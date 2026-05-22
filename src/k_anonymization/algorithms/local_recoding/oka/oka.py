import random
from functools import partial

import torch
from numpy import argmin, array
from tqdm import tqdm

from k_anonymization.core import Dataset, Parallel

from .._utils import get_information_loss
from ..local_recoding_algorithm import (
    GroupAnonymization,
    GroupAnonymizationBuiltIn,
    LocalRecodingAlgorithm,
)
from ._utils import oka_get_distance_parallel, oka_init_cluster

BAR_FORMAT = "{desc}: {percentage:6.2f}% |{bar}| [{elapsed}, {rate_fmt}]"


class OKA(LocalRecodingAlgorithm):
    """
    Implementation of the One-Pass K-Means (OKA) clustering algorithm.

    OKA adopts the idea of the K-Means clustering algorithm. It initiates
    all clusters (groups) of records at once, each with a random seed,
    and distributes the remaining records individually to them
    based on minimal clusters' information losses. Then, OKA
    performs a one-time adjustment step, where furthest records
    in clusters of size > `k` (subject to distance to centroid) are
    picked out and redistributed to those of size < `k`, until every
    cluster contains at least `k` records.

    Parameters
    ----------
    dataset : Dataset
        The Dataset object holding the original data and its metadata.
    k : int
        The privacy parameter `k`.
    group_anonymization : GroupAnonymization
        The method to anonymize the resulting clusters after applying
        local recoding.
        It is possible to use an example method in
        ``GroupAnonymizationBuiltIn``, or create a custom method
        ``custom_group_anonymization(group: list, props: Any) -> list``.
        Default: ``GroupAnonymizationBuiltIn.SUMMARIZATION``
    seed : int
        Random seed for the initial record selection to ensure reproducibility.
    parallel : bool
        Boolean flag to enable parallel processing.
    cpu_cores : int
        The number of CPU cores to utilize when ``parallel`` is True.

    Attributes
    ----------
    is_parallel : bool
        Whether the algorithm is running in parallel mode.
    information_loss : float
        The total information loss calculated across all clusters.
    rand_idx : list
        The indices of the records randomly selected to serve as
        initial cluster seeds.
    """

    def __init__(
        self,
        dataset: Dataset,
        k: int,
        group_anonymization: GroupAnonymization = GroupAnonymizationBuiltIn.SUMMARIZATION,
        seed: int = None,
        device: str = "cpu",
        cpu_cores: int = Parallel.max_cores - 1,
    ):
        super().__init__(dataset, k, group_anonymization)
        self.seed = seed
        self.cpu_cores = cpu_cores

        _available_devices = ["cpu"]
        if torch.cuda.is_available():
            _available_devices.append("cuda")
        elif torch.backends.mps.is_available():
            _available_devices.append("mps")

        if device in _available_devices:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")
            print(
                f"This machine only has these devices available: {', '.join(_available_devices)}"
            )
            print("Fall back to cpu.")

        self._prepare_tensors()

    def _prepare_tensors(self):
        data = self.org_data.values
        self.data_size = data.shape[0]

        _qids_idx_num = self.dataset.qids_idx_numerical
        _qids_idx_cat = self.dataset.qids_idx_categorical

        # Build Numerical tensor
        self.ts_num = None
        if _qids_idx_num:
            self.ts_num = torch.tensor(
                data[:, _qids_idx_num].astype(float),
                dtype=torch.float32,
                device=self.device,
            )
            # Safely normalize to prevent Division by Zero
            ranges = self.ts_num.amax(dim=0) - self.ts_num.amin(dim=0)
            ranges[ranges == 0] = 1.0
            self.ts_num /= ranges

        # Build Categorical tensor and distance matrices
        self.ts_cat = None
        self.ts_cat_h1 = None
        if _qids_idx_cat:
            dist_mat_cat = []
            cat_data = []
            cat_h1_data = []
            for idx in _qids_idx_cat:
                hierarchy = self.hierarchies[idx]
                values = hierarchy.leaves
                height = hierarchy.height

                values_mapping = {v: i for i, v in enumerate(values)}

                # Separate cat_qids with height==1 to reduce zero-padding size
                if height == 1:
                    cat_h1_data.append(
                        [values_mapping[val] for val in data[:, idx].tolist()]
                    )
                else:
                    cat_data.append(
                        [values_mapping[val] for val in data[:, idx].tolist()]
                    )

                    dist_mat = torch.full(
                        [len(values), len(values)],
                        height,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    hierarchy_df = hierarchy.hierarchy_df.copy()
                    hierarchy_df[0] = hierarchy_df[0].apply(lambda x: values_mapping[x])

                    hierarchy_df = hierarchy_df.groupby(
                        list(range(1, height)), as_index=False
                    )[0].agg(list)

                    for group in hierarchy_df[0]:
                        group = array(group)
                        dist_mat[group[:, None], group] = 1

                    for level in range(2, height):
                        for v in hierarchy_df[level].unique():
                            children = hierarchy_df[hierarchy_df[level] == v][0]
                            for i, group_i in enumerate(children[:-1]):
                                group_i = array(group_i)[:, None]
                                for group_j in children[i + 1 :]:
                                    dist_mat[group_i, group_j] = level
                                    dist_mat[group_j, group_i] = level

                    dist_mat /= height
                    dist_mat.fill_diagonal_(0)

                    dist_mat_cat.append(dist_mat)

            max_mat_size = max(mat.shape[0] for mat in dist_mat_cat)
            self.dist_mat_cat = torch.zeros(
                (len(cat_data), max_mat_size, max_mat_size), device=self.device
            )
            for idx, mat in enumerate(dist_mat_cat):
                mat_size = mat.shape[0]
                self.dist_mat_cat[idx, :mat_size, :mat_size] = mat

            if len(cat_data) > 0:
                self.ts_cat = torch.tensor(
                    cat_data, dtype=torch.long, device=self.device
                ).T

            if len(cat_h1_data) > 0:
                self.ts_cat_h1 = torch.tensor(
                    cat_h1_data, dtype=torch.long, device=self.device
                ).T

    def do_local_recoding(self):
        """
        Perform the OKA clustering algorithm entirely using GPU tensors.
        """
        # Number of cluster
        self.clusters_count = self.data_size // self.k
        ts_clusters_idx = torch.arange(self.clusters_count, device=self.device)

        # Generate first record (seed) for each cluster
        random.seed(self.seed)
        self.rand_idx = random.sample(range(self.data_size), self.clusters_count)

        has_num_qids = self.ts_num is not None
        has_cat_qids = self.ts_cat is not None
        has_cat_h1_qids = self.ts_cat_h1 is not None

        # Current size of each cluster
        # Initialize to 1 because they all contain 1 record (the seed)
        cluster_sizes = torch.ones(
            self.clusters_count, dtype=torch.float32, device=self.device
        )
        # The mean value of each numerical attribute,
        # representing the value at the centroid.
        if self.ts_num is not None:
            centroids_num = self.ts_num[self.rand_idx].clone()

        # The distance matrices from every cluster to every possible cat values
        # Shape: (clusters_count, cat_qids_count, max_mat_size)
        # This is for calculating the categorical distance from a record
        # to the centroid of every cluster.
        # It must be updated every time a record is added to a cluster.
        if self.ts_cat is not None:
            ts_cat_qids_idx = torch.arange(self.ts_cat.shape[1], device=self.device)
            cluster_cat_dist = self.dist_mat_cat[
                ts_cat_qids_idx.unsqueeze(0),
                self.ts_cat[self.rand_idx],
            ]
        # For categorical qids with height == 1, distance is 0 for the same value
        # and 1 otherwise. In other words, distance is 0 if the cluster contains
        # identical values and they are the same with the reference value.
        # This tensor stores whether or not the cluster contains identical values.
        # It must be updated every time a record is added to a cluster.
        if self.ts_cat_h1 is not None:
            cluster_cat_h1_dist = torch.zeros(
                (self.clusters_count, self.ts_cat_h1.shape[1]),
                dtype=torch.bool,
                device=self.device,
            )
            # The categorical values of the seed
            # If the cluster stores identical values, all values are the same
            # with the seed.
            seed_cat_h1 = self.ts_cat_h1[self.rand_idx]

        # The index of the cluster that a record is assigned
        cluster_assignments = torch.full(
            (self.data_size,), -1, dtype=torch.long, device=self.device
        )
        cluster_assignments[self.rand_idx] = torch.arange(
            self.clusters_count, device=self.device
        )

        def put_record_to_suitable_cluster(r_idx, considered_clusters=None):
            r_num = self.ts_num[r_idx] if has_num_qids else None
            r_cat = self.ts_cat[r_idx] if has_cat_qids else None
            r_cat_h1 = self.ts_cat_h1[r_idx] if has_cat_h1_qids is not None else None

            # If considered_clusters is not specified, calculate distance to all clusters
            # Use if-else to prevent slowing down due to indexing
            has_considered_clusters = considered_clusters is not None

            _sizes = (
                cluster_sizes[considered_clusters]
                if has_considered_clusters
                else cluster_sizes
            )

            num_dist = (
                torch.sum(
                    torch.abs(
                        (
                            centroids_num[considered_clusters]
                            if has_considered_clusters
                            else centroids_num
                        )
                        - r_num
                    ),
                    dim=-1,
                )
                if has_num_qids
                else 0
            )
            cat_dist = (
                cluster_cat_dist[
                    (
                        considered_clusters.unsqueeze(1)
                        if has_considered_clusters
                        else ts_clusters_idx.unsqueeze(1)
                    ),
                    ts_cat_qids_idx.unsqueeze(0),
                    r_cat.unsqueeze(0),
                ].sum(dim=1)
                if has_cat_qids
                else 0
            )
            cat_h1_dist = (
                torch.logical_or(
                    (
                        cluster_cat_h1_dist[considered_clusters]
                        if has_considered_clusters
                        else cluster_cat_h1_dist
                    ),
                    r_cat_h1
                    != (
                        seed_cat_h1[considered_clusters]
                        if has_considered_clusters
                        else seed_cat_h1
                    ),
                ).sum(dim=1, dtype=torch.float32)
                if has_cat_h1_qids
                else 0
            )
            total_dist = _sizes * (num_dist + cat_dist + cat_h1_dist)
            best_local_idx = torch.argmin(total_dist).item()
            best_cluster_idx = (
                considered_clusters[best_local_idx].item()
                if has_considered_clusters
                else best_local_idx
            )

            cluster_sizes[best_cluster_idx] += 1

            # New mean = ( prev_mean * (new_size - 1) + r_num ) / new_size
            if has_num_qids:
                centroids_num[best_cluster_idx] = (
                    centroids_num[best_cluster_idx]
                    * (
                        (cluster_sizes[best_cluster_idx] - 1)
                        / cluster_sizes[best_cluster_idx]
                    )
                    + r_num / cluster_sizes[best_cluster_idx]
                )
            # New categorical distance to all records is the maximum between
            # categorical distance before adding r_cat to all records, and
            # categorical distance between r_cat to all records
            if has_cat_qids:
                cluster_cat_dist[best_cluster_idx] = torch.max(
                    cluster_cat_dist[best_cluster_idx],
                    self.dist_mat_cat[ts_cat_qids_idx, r_cat, :],
                )
            if has_cat_h1_qids:
                cluster_cat_h1_dist[best_cluster_idx] = torch.logical_or(
                    cluster_cat_h1_dist[best_cluster_idx],
                    r_cat_h1 != seed_cat_h1[best_cluster_idx],
                )

            cluster_assignments[r_idx] = best_cluster_idx

        # ==========================================
        # 1. Clustering Stage
        # ==========================================

        for r_idx in tqdm(
            range(len(cluster_assignments)),
            desc="  Clustering",
            bar_format=BAR_FORMAT,
        ):
            if cluster_assignments[r_idx] != -1:
                continue

            put_record_to_suitable_cluster(r_idx)

        # ==========================================
        # 2. Adjustment Stage
        # ==========================================
        more_than_k = (cluster_sizes > self.k).nonzero(as_tuple=True)[0]
        adjusting_pool = []

        # Find and sort excess records across clusters of size > k
        for idx in tqdm(
            range(len(more_than_k)),
            desc="    Trimming",
            bar_format=BAR_FORMAT,
        ):
            cluster_idx = more_than_k[idx].item()
            members_idx = (cluster_assignments == cluster_idx).nonzero(as_tuple=True)[0]

            cluster_r_num = self.ts_num[members_idx] if has_num_qids else None
            cluster_r_cat = self.ts_cat[members_idx] if has_cat_qids else None

            num_dist = (
                torch.sum(
                    torch.abs(
                        centroids_num[cluster_idx : cluster_idx + 1] - cluster_r_num
                    ),
                    dim=-1,
                )
                if has_num_qids
                else 0
            )
            cat_dist = (
                cluster_cat_dist[
                    cluster_idx,
                    ts_cat_qids_idx.unsqueeze(0),
                    cluster_r_cat,
                ].sum(dim=1)
                if has_cat_qids
                else 0
            )
            cat_h1_dist = (
                cluster_cat_h1_dist[cluster_idx].sum(dtype=torch.float32).item()
                if has_cat_h1_qids
                else 0
            )

            total_dist = cluster_sizes[cluster_idx] * (
                num_dist + cat_dist + cat_h1_dist
            )

            # Remove the furthest records to centroid so that cluster size = k
            sorted_indices = torch.argsort(total_dist, descending=True)
            removal_count = len(members_idx) - self.k
            removed_indices = members_idx[sorted_indices[:removal_count]]
            kept_indices = members_idx[sorted_indices[removal_count:]]

            adjusting_pool.extend(removed_indices.tolist())
            cluster_assignments[removed_indices] = -1

            # Recalculate size, centroids_num, and cat_distance from remaining members
            cluster_sizes[cluster_idx] = self.k
            if has_num_qids:
                centroids_num[cluster_idx] = self.ts_num[kept_indices].mean(dim=0)
            if has_cat_qids:
                cluster_cat_dist[cluster_idx] = self.dist_mat_cat[
                    ts_cat_qids_idx.unsqueeze(1),
                    self.ts_cat[kept_indices].T,
                    :,
                ].amax(dim=1)
            if has_cat_h1_qids:
                cluster_cat_h1_dist[cluster_idx] = (
                    self.ts_cat_h1[kept_indices] != seed_cat_h1[cluster_idx]
                ).any(dim=0)

        for idx in tqdm(
            range(len(adjusting_pool)),
            desc="Reallocating",
            bar_format=BAR_FORMAT,
        ):
            r_idx = adjusting_pool[idx]

            # Limit to clusters of size <k if available
            less_than_k = (cluster_sizes < self.k).nonzero(as_tuple=True)[0]
            if len(less_than_k) > 0:
                put_record_to_suitable_cluster(r_idx, less_than_k)
            else:
                put_record_to_suitable_cluster(r_idx)

        # ==========================================
        # 3. Clean Re-Construction of Output
        # ==========================================
        full_data_list = self.anon_data.values

        clusters = []
        for cluster_idx in range(self.clusters_count):
            members_idx = (
                (cluster_assignments == cluster_idx).nonzero(as_tuple=True)[0].tolist()
            )
            clusters.append(full_data_list[members_idx].tolist())

        return clusters


class OKAUnOptimized(LocalRecodingAlgorithm):
    """
    Implementation of the One-Pass K-Means (OKA) clustering algorithm.

    OKA adopts the idea of the K-Means clustering algorithm. It initiates
    all clusters (groups) of records at once, each with a random seed,
    and distributes the remaining records individually to them
    based on minimal clusters' information losses. Then, OKA
    performs a one-time adjustment step, where furthest records
    in clusters of size > `k` (subject to distance to centroid) are
    picked out and redistributed to those of size < `k`, until every
    cluster contains at least `k` records.

    Parameters
    ----------
    dataset : Dataset
        The Dataset object holding the original data and its metadata.
    k : int
        The privacy parameter `k`.
    group_anonymization : GroupAnonymization
        The method to anonymize the resulting clusters after applying
        local recoding.
        It is possible to use an example method in
        ``GroupAnonymizationBuiltIn``, or create a custom method
        ``custom_group_anonymization(group: list, props: Any) -> list``.
        Default: ``GroupAnonymizationBuiltIn.SUMMARIZATION``
    seed : int
        Random seed for the initial record selection to ensure reproducibility.
    parallel : bool
        Boolean flag to enable parallel processing.
    cpu_cores : int
        The number of CPU cores to utilize when ``parallel`` is True.

    Attributes
    ----------
    is_parallel : bool
        Whether the algorithm is running in parallel mode.
    information_loss : float
        The total information loss calculated across all clusters.
    rand_idx : list
        The indices of the records randomly selected to serve as
        initial cluster seeds.
    """

    def __init__(
        self,
        dataset: Dataset,
        k: int,
        group_anonymization: GroupAnonymization = GroupAnonymizationBuiltIn.SUMMARIZATION,
        seed: int = None,
        parallel: bool = False,
        cpu_cores: int = Parallel.max_cores - 1,
    ):
        """
        Initialize the OKA algorithm.

        Parameters
        ----------
        dataset : Dataset
            The Dataset object holding the original data and its metadata.
        k : int
            The privacy parameter `k`.
        group_anonymization : GroupAnonymization
            The method to anonymize the resulting clusters after applying
            local recoding.
            It is possible to use an example method in
            ``GroupAnonymizationBuiltIn``, or create a custom method
            ``custom_group_anonymization(group: list, props: Any) -> list``.
            Default: ``GroupAnonymizationBuiltIn.SUMMARIZATION``
        seed : int
            Random seed for the initial record selection to ensure reproducibility.
        parallel : bool
            Boolean flag to enable parallel processing.
        cpu_cores : int
            The number of CPU cores to utilize when ``parallel`` is True.
        """
        super().__init__(dataset, k, group_anonymization)
        self.seed = seed
        self.cpu_cores = cpu_cores
        self.is_parallel = parallel
        self.__parallel = Parallel(cpu_cores)
        # Sets up partial functions for distance calculations and cluster
        # initialization, which are crucial for parallel processing.
        self.__get_distance_parallel = partial(oka_get_distance_parallel)
        self.__init_cluster = partial(
            oka_init_cluster,
            qids_idx=self.qids_idx,
            is_categorical=self.is_categorical,
            max_ranges=self.max_ranges,
            hierarchies=self.hierarchies,
        )

    def init_clusters(self):
        """
        Initialize all clusters, each with a random record.

        The number of seeds is calculated as ``round_down(D/k)``,
        where `D` is the total number of records.

        Returns
        -------
        list
            A list of initialized cluster objects.
        """
        random.seed(self.seed)
        self.rand_idx = random.sample(
            range(0, self.anon_data.shape[0]),
            int(self.anon_data.shape[0] / self.k),
        )
        rand_records = [self.anon_data.loc[i].tolist() for i in self.rand_idx]
        return (
            self.__parallel.perform(self.__init_cluster, rand_records)
            if self.is_parallel
            else [self.__init_cluster(r) for r in rand_records]
        )

    def find_best_cluster(self, record: list, clusters: list):
        """
        Find the closest cluster centroid for a given record.

        Calculates the distance between the given record and all current
        cluster centroids to find the best similar cluster.

        Parameters
        ----------
        record : list
            The record to be assigned.
        clusters : list[list]
            The list of clusters.

        Returns
        -------
        int
            The index of the cluster with the minimum distance.
        """
        f = partial(self.__get_distance_parallel, record=record)
        distances = (
            self.__parallel.perform(f, clusters)
            if self.is_parallel
            else [cluster.distance(record) for cluster in clusters]
        )

        best_idx = argmin(distances).item()

        return best_idx

    def get_adjusting_records(self, clusters: list):
        """
        Extract excess records from clusters that exceed size `k`.

        Used during the adjustment stage to free up records that can be
        reassigned to clusters that haven't yet `k`-anonymous.

        Parameters
        ----------
        clusters
            The list of clusters with more than k members.

        Returns
        -------
        list
            A list of records removed from the provided clusters.
        """

        def __get_adjusting_records(cluster, k):
            cluster.sort_by_distance()
            return cluster.remove([0, len(cluster) - k])

        _adjusting_records = [__get_adjusting_records(c, self.k) for c in clusters]
        return sum(_adjusting_records, [])

    def do_local_recoding(self):
        """
        Perform the OKA clustering algorithm.

        The workflow consists of:

        1. Initialize clusters with random seeds.

        2. Clustering stage: Assign every record to the closest cluster
           based on distance to centroid.

        3. Adjustment stage: Rebalance records from "over-full" clusters (`> k`)
           to "under-full" clusters (`< k`) to ensure all clusters are valid.

        Returns
        -------
        list
            The final list of clusters.
        """
        if self.is_parallel:
            print(f"Parallelize with {self.cpu_cores} core(s).")
            self.__parallel.activate()

        clusters = self.init_clusters()

        self.anon_data.drop(self.rand_idx, inplace=True)
        data = self.anon_data.values.tolist()

        clustering_progress_bar = tqdm(
            total=len(data),
            desc="   Clustering Progress",
            bar_format=BAR_FORMAT,
        )

        # Clustering Stage
        while len(data) > 0:
            record = data.pop()
            best_cluster_idx = self.find_best_cluster(record, clusters)
            clusters[best_cluster_idx].add(record)
            clustering_progress_bar.update(1)

        clustering_progress_bar.close()
        adjustment_progress_bar = tqdm(
            total=(len(clusters) * 2),
            desc="   Adjustment Progress",
            bar_format=BAR_FORMAT,
        )

        # Adjustment Stage
        less_than_k_clusters = []
        more_than_k_clusters = []
        for cluster in clusters:
            adjustment_progress_bar.update(1)
            if len(cluster) == self.k:
                continue
            elif len(cluster) < self.k:
                less_than_k_clusters.append(cluster)
            else:
                more_than_k_clusters.append(cluster)
        adjusting_records = self.get_adjusting_records(more_than_k_clusters)

        while len(adjusting_records) > 0:
            record = adjusting_records.pop()
            if len(less_than_k_clusters) > 0:
                best_cluster_idx = self.find_best_cluster(record, less_than_k_clusters)
                less_than_k_clusters[best_cluster_idx].add(record)
                if len(less_than_k_clusters[best_cluster_idx]) >= self.k:
                    less_than_k_clusters.pop(best_cluster_idx)
            else:
                best_cluster_idx = self.find_best_cluster(record, clusters)
                clusters[best_cluster_idx].add(record)

        adjustment_progress_bar.update(len(clusters))
        adjustment_progress_bar.close()

        self.information_loss = 0
        for cluster in clusters:
            self.information_loss += get_information_loss(
                None,
                cluster.member,
                self.qids_idx,
                self.is_categorical,
                self.max_ranges,
                self.hierarchies,
            )

        return clusters
