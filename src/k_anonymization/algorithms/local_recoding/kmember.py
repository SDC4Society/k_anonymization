import random
from functools import partial

import torch
from numpy import argmax, argmin, array
from tqdm.auto import tqdm

from k_anonymization.core import Dataset, Parallel

from ._utils import get_distance, get_information_loss
from .local_recoding_algorithm import (
    GroupAnonymization,
    GroupAnonymizationBuiltIn,
    LocalRecodingAlgorithm,
)

try:
    __IPYTHON__  # type: ignore # noqa: F821
    _bar_format = None
except:
    _bar_format = "{l_bar}{bar:20}|{n_fmt}/{total_fmt} [{elapsed}]"


class KMember(LocalRecodingAlgorithm):
    """
    Implementation of the K-Member clustering algorithm.

    K-Member greedily constructs one cluster (group) of records at a time
    until the whole dataset is divided into groups of at least `k` records.
    It initiates the first cluster by randomly selecting an initial record
    (seed), then finds and adds `k-1` other records that minimizes the
    cluster's information loss.
    From the second cluster onward, it picks a new seed which is the furthest
    from the previous seed, and repeats the record selection process until
    there are less than `k` records remaining.
    Finally, each remaining record is added to one of the existing clusters
    that minimizes the cluster's information loss.

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
    device : str
        Computing device, e.g., cpu, cuda, mps.
        Default: ``"cpu"``
    cpu_cores : int
        The number of CPU cores to utilize running on ``device="cpu"``.
        Default: 4

    Attributes
    ----------
    information_loss : float
        The total information loss calculated across all clusters.

    See Also
    --------
    k_anonymization.core.Parallel :
        Utility wrapper for paralellizing tasks across multiple CPU cores.
    """

    def __init__(
        self,
        dataset: Dataset,
        k: int,
        group_anonymization: GroupAnonymization = GroupAnonymizationBuiltIn.SUMMARIZATION,
        seed: int = None,
        device: str = "cpu",
        cpu_cores: int = 4,
    ):
        """
        Initialize the KMember algorithm.

        Parameters
        ----------
        dataset : Dataset
            The Dataset object holding the original data and its metadata.
        k : int
            The privacy parameter 'k'.
        group_anonymization : GroupAnonymization
            The method to anonymize the resulting clusters after applying
            local recoding.
            It is possible to use an example method in
            ``GroupAnonymizationBuiltIn``, or create a custom method
            ``custom_group_anonymization(group: list, props: Any) -> list``.
            Default: ``GroupAnonymizationBuiltIn.SUMMARIZATION``
        seed : int
            Random seed for the initial record selection to ensure reproducibility.
        device : str
            Computing device, e.g., cpu, cuda, mps.
            Default: ``"cpu"``
        cpu_cores : int
            The number of CPU cores to utilize running on ``device="cpu"``.
            Default: 4
        """
        super().__init__(dataset, k, group_anonymization)
        self.seed = seed
        self.cpu_cores = cpu_cores

        _available_devices = ["cpu"]
        if torch.cuda.is_available():
            _available_devices.append("cuda")
        elif torch.mps.is_available():
            _available_devices.append("mps")

        if device in _available_devices:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")
            print(f"This machine only has these devices available: {
                    ", ".join(_available_devices)
                    }")
            print("Fall back to cpu.")

        self._prepare_tensors()

    def _prepare_tensors(self):

        data = self.org_data.values
        self.data_size = data.shape[0]

        _qids_idx_num = self.dataset.qids_idx_numerical
        _qids_idx_cat = self.dataset.qids_idx_categorical

        # Build Numerical tensor
        if _qids_idx_num:
            self.tensor_num = torch.tensor(
                data[:, _qids_idx_num].astype(float),
                dtype=torch.float32,
                device=self.device,
            )
            self.tensor_num /= self.tensor_num.amax(dim=0) - self.tensor_num.amin(dim=0)
        else:
            self.tensor_num = torch.empty((len(data), 0), device=self.device)

        # Build Categorical tensor and distance matrices
        self.distance_matrices_cat = []
        if _qids_idx_cat:
            cat_data = []
            for idx in _qids_idx_cat:
                hierarchy = self.hierarchies[idx]
                values = hierarchy.leaves
                height = hierarchy.height

                values_mapping = {v: i for i, v in enumerate(values)}
                cat_data.append([values_mapping[val] for val in data[:, idx].tolist()])
                if height == 1:
                    distance_matrix = None
                else:
                    distance_matrix = torch.zeros(
                        [len(values), len(values)],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    distance_matrix += height
                    hierarchy_df = hierarchy.hierarchy_df.copy()
                    hierarchy_df[0] = hierarchy_df[0].apply(lambda x: values_mapping[x])

                    hierarchy_df = hierarchy_df.groupby(
                        list(range(1, height)), as_index=False
                    )[0].agg(list)

                    for group in hierarchy_df[0]:
                        group = array(group)
                        distance_matrix[group[:, None], group] = 1

                    for level in range(2, height):
                        for v in hierarchy_df[level].unique():
                            children = hierarchy_df[hierarchy_df[level] == v][0]
                            for i, group_i in enumerate(children[:-1]):
                                group_i = array(group_i)[:, None]
                                for group_j in children[i + 1 :]:
                                    distance_matrix[group_i, group_j] = level
                                    distance_matrix[group_j, group_i] = level

                    distance_matrix /= height
                    distance_matrix.fill_diagonal_(0)

                self.distance_matrices_cat.append(distance_matrix)

            self.tensor_cat = torch.tensor(
                cat_data, dtype=torch.int, device=self.device
            ).T
        else:
            self.tensor_cat = torch.empty(
                (len(data), 0), dtype=torch.int, device=self.device
            )

    def find_furthest_record_from_r(
        self, r_idx: int, is_picked: torch.Tensor | None = None
    ):
        """
        Find the most distant record from the given record index `r_idx`.

        This is used to find a "seed" for a new cluster that is far
        away from the previously processed cluster.

        Parameters
        ----------
        r_idx : int
            Index of the given record.
        is_picked : torch.Tensor | None
            1-Dimensional tensor of boolen flags indicating whether the record
            at each index has been picked out from the data (``shape=[len(org.data)]``).
            Default: None, a.k.a., all records have not been picked.

        Returns
        -------
        int
            Index of the furthest record.
        """
        r_num = self.tensor_num[r_idx : r_idx + 1]
        r_cat = self.tensor_cat[r_idx]

        # Compute Numerical Distances (Broadcasted)
        distance_num = torch.sum(torch.abs(self.tensor_num - r_num), dim=1)

        # Compute Categorical Distances via Matrix Lookups (Broadcasted)
        distance_cat = torch.zeros_like(distance_num)
        for i, matrix in enumerate(self.distance_matrices_cat):
            # Fetch distance from all categorical values in column i to r_cat[i]
            if matrix is None:
                distance_cat += (self.tensor_cat[:, i] != r_cat[i]).float()
            else:
                distance_cat += matrix[self.tensor_cat[:, i], r_cat[i]]

        distance = distance_num + distance_cat

        # Ignore records that are already picked
        if is_picked is not None:
            distance[is_picked] = -1.0

        furthest_idx = torch.argmax(distance).item()
        return furthest_idx

    def find_best_record_for_cluster(
        self,
        cluster_r_idxs,
        is_picked: torch.Tensor | None = None,
    ):
        """
        Find the record that minimizes the information loss of the given cluster.

        Iterates through available records and calculates the potential
        increase in information loss if each record were added to the
        given cluster, then pick the one that causes the lowest loss.

        Parameters
        ----------
        cluster_r_idxs : list[int] | torch.Tensor
            The list of indices of the records in the given cluster.
        is_picked : torch.Tensor | None
            1-Dimensional tensor of boolen flags indicating whether the record
            at each index has been picked out from the data (``shape=[len(org.data)]``).
            Default: None, a.k.a., all records have not been picked.

        Returns
        -------
        tuple[int, float]
            (Index of the best record, the resulting information loss).
        """
        cluster_size = len(cluster_r_idxs)

        cluster_num = self.tensor_num[cluster_r_idxs]
        cluster_cat = self.tensor_cat[cluster_r_idxs]

        # --- 1. Vectorized Numerical Loss ---
        if cluster_num.shape[1] > 0:
            # The max/min bounds after adding each record:
            cluster_num_max = torch.maximum(self.tensor_num, cluster_num.amax(dim=0))
            cluster_num_min = torch.minimum(self.tensor_num, cluster_num.amin(dim=0))

            iloss_num = torch.sum(cluster_num_max - cluster_num_min, dim=1)
        else:
            iloss_num = torch.zeros(self.tensor_num.shape[0], device=self.device)

        # --- 2. Vectorized Categorical Loss ---
        iloss_cat = torch.zeros_like(iloss_num)

        for i, matrix in enumerate(self.distance_matrices_cat):
            # Information loss of each categorical attribute in a group is
            # the normalized LCA height of its values, which is equivalent to
            # maximum pairwise distance between any two values in the group.

            # Information loss of the considered attribute from all records
            # to cluster's records
            if matrix is None:
                iloss_all_r_to_cluster = (
                    (self.tensor_cat[:, i] != cluster_cat[:, i].unsqueeze(1))
                    .float()
                    .amax(dim=0)
                )
            else:
                iloss_all_r_to_cluster = matrix[self.tensor_cat[:, i]][
                    :, cluster_cat[:, i]
                ].amax(dim=1)

            if cluster_size > 1:
                # If cluster has more than 1 record, get the maximum between
                # the local information loss incurred by its own records, and
                # information loss incurred by adding each new record.
                if matrix is None:
                    if cluster_cat[:, i].unique(sorted=False).size()[0] > 1:
                        iloss_cat += torch.ones_like(iloss_num)
                    else:
                        iloss_cat += iloss_all_r_to_cluster
                else:
                    iloss_local = matrix[cluster_cat[:, i]][:, cluster_cat[:, i]].max()
                    iloss_cat += torch.maximum(iloss_all_r_to_cluster, iloss_local)
            else:
                iloss_cat += iloss_all_r_to_cluster

        information_loss = (cluster_size + 1) * (iloss_num + iloss_cat)

        # Ignore already clustered records
        if is_picked is not None:
            information_loss[is_picked] = float("inf")

        best_idx = torch.argmin(information_loss).item()
        return (best_idx, information_loss[best_idx].item())

    def find_best_cluster_for_record(
        self, all_clusters_r_idxs: torch.Tensor, r_idx: int, pos: int
    ):
        """
        Assign an orphaned record to the most compatible existing cluster.

        Used at the end of the process to assign any remaining records
        to existing clusters while minimizing added information loss.

        Parameters
        ----------
        all_clusters_r_idxs : torch.Tensor
            List of indices of records in already formed clusters.
        r_idx : int
            Index of the orphaned record to be considered.
        pos : int
            Assumed index of the considered record in the cluster.

        Returns
        -------
        tuple[int, float]
            (index of the best cluster, the resulting information loss).
        """
        all_clusters_r_idxs[:, pos] = r_idx
        clusters_sizes = (
            all_clusters_r_idxs != all_clusters_r_idxs.unsqueeze(-1)[:, 0]
        ).sum(dim=1) + 1

        all_clusters_num = self.tensor_num[all_clusters_r_idxs]
        all_clusters_cat = self.tensor_cat[all_clusters_r_idxs]

        iloss_num = (
            (all_clusters_num.amax(dim=1) - all_clusters_num.amin(dim=1)).sum(dim=1)
            if all_clusters_num.shape[1] > 0
            else torch.zeros(
                self.number_of_clusters, dtype=torch.float32, device=self.device
            )
        )

        iloss_cat = torch.zeros(
            self.number_of_clusters, dtype=torch.float32, device=self.device
        )
        for i, matrix in enumerate(self.distance_matrices_cat):
            this_cat = all_clusters_cat[:, :, i]
            if matrix is None:
                iloss_cat += (
                    (this_cat != this_cat.unsqueeze(-1)[:, 0]).sum(dim=1) > 0
                ).float()
            else:
                iloss_cat += (
                    matrix[this_cat.unsqueeze(-1), this_cat.unsqueeze(1)]
                    .flatten(1)
                    .amax(dim=1)
                )

        information_loss = clusters_sizes * (iloss_num + iloss_cat)
        best_cluster_idx = information_loss.argmin()

        all_clusters_r_idxs[:, pos] = all_clusters_r_idxs[:, 0]
        return (best_cluster_idx, information_loss[best_cluster_idx].item())

    def do_local_recoding(self):
        """
        Perform the K-Member clustering algorithm.

        The workflow consists of:

        1. Pick a seed record (a random record for the 1st iteration,
           the furthest record from the previous seed otherwise).

        2. Build a cluster of size k by greedily adding records that
           minimize information loss.

        3. Repeat 1-3 until fewer than k records remain.

        4. Distribute remaining records to the most suitable existing
           clusters.

        Returns
        -------
        list
            The final list of clusters.
        """
        data = self.anon_data.values.tolist()
        self.number_of_clusters = self.data_size // self.k

        if self.device.type == "cpu":
            torch.set_num_threads(self.cpu_cores)

        all_clusters_r_idxs = torch.zeros(
            (self.number_of_clusters, self.k + self.data_size % self.k),
            dtype=torch.int,
            device=self.device,
        )
        information_losses = []
        r_i_idx = None

        is_picked = torch.zeros(self.data_size, dtype=torch.bool, device=self.device)

        progress_bar = tqdm(
            total=self.data_size,
            desc="   Clustering Progress",
            bar_format=_bar_format,
        )

        for this_cluster_r_idxs in all_clusters_r_idxs:
            if r_i_idx is None:
                random.seed(self.seed)
                r_i_idx = random.randrange(self.data_size)
            else:
                r_i_idx = self.find_furthest_record_from_r(r_i_idx, is_picked)
            is_picked[r_i_idx] = True
            this_cluster_r_idxs[0] = r_i_idx
            this_cluster_r_idxs[self.k :] = r_i_idx

            for i in range(1, self.k):
                r_j_idx, this_information_loss = self.find_best_record_for_cluster(
                    this_cluster_r_idxs[:i], is_picked
                )
                is_picked[r_j_idx] = True
                this_cluster_r_idxs[i] = r_j_idx

            information_losses.append(this_information_loss)
            progress_bar.update(self.k)

        orphaned_idxs = torch.nonzero(~is_picked).flatten(0).tolist()
        for pos, r_idx in enumerate(orphaned_idxs):
            best_cluster_idx, new_information_loss = self.find_best_cluster_for_record(
                all_clusters_r_idxs, r_idx, pos + self.k
            )
            information_losses[best_cluster_idx] = new_information_loss
            all_clusters_r_idxs[best_cluster_idx, pos + self.k] = r_idx
            progress_bar.update(1)

        self.information_loss = sum(information_losses)
        progress_bar.close()

        clusters = []
        for cluster_idxs in all_clusters_r_idxs.tolist():
            clusters.append([data[idx] for idx in sorted(set(cluster_idxs))])

        return clusters


class KMemberUnOptimized(LocalRecodingAlgorithm):
    """
    Implementation of the K-Member clustering algorithm.

    K-Member greedily constructs one cluster (group) of records at a time
    until the whole dataset is divided into groups of at least `k` records.
    It initiates the first cluster by randomly selecting an initial record
    (seed), then finds and adds `k-1` other records that minimizes the
    cluster's information loss.
    From the second cluster onward, it picks a new seed which is the furthest
    from the previous seed, and repeats the record selection process until
    there are less than `k` records remaining.
    Finally, each remaining record is added to one of the existing clusters
    that minimizes the cluster's information loss.

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
    seed
        Random seed for the initial record selection to ensure reproducibility.
    parallel
        Boolean flag to enable parallel processing.
    cpu_cores
        The number of CPU cores to utilize when ``parallel`` is True.

    Attributes
    ----------
    is_parallel : bool
        Whether the algorithm is running in parallel mode.
    information_loss : float
        The total information loss calculated across all clusters.

    See Also
    --------
    k_anonymization.core.Parallel :
        Utility wrapper for paralellizing tasks across multiple CPU cores.
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
        Initialize the KMember algorithm.

        Parameters
        ----------
        dataset : Dataset
            The Dataset object holding the original data and its metadata.
        k : int
            The privacy parameter 'k'.
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
        # Sets up partial functions for distance and information loss
        # calculations, which are crucial for parallel processing.
        self.get_distance = partial(
            get_distance,
            qids_idx=self.qids_idx,
            is_categorical=self.is_categorical,
            max_ranges=self.max_ranges,
            hierarchies=self.hierarchies,
        )
        self.get_information_loss = partial(
            get_information_loss,
            qids_idx=self.qids_idx,
            is_categorical=self.is_categorical,
            max_ranges=self.max_ranges,
            hierarchies=self.hierarchies,
        )

    def find_furthest_record_from_r(self, r: list, data: list):
        """
        Find the most distant record from the given record `r`.

        This is used to find a "seed" for a new cluster that is far
        away from the previously processed cluster.

        Parameters
        ----------
        r : list
            The given record.
        data : list[list]
            The list of available records.

        Returns
        -------
        tuple
            (the furthest record, its index).
        """
        f = partial(self.get_distance, record=r)
        distances = (
            self.__parallel.perform(f, data)
            if self.is_parallel
            else [f(record) for record in data]
        )
        furthest_idx = argmax(distances).item()
        return (data[furthest_idx], furthest_idx)

    def find_best_record(self, data, cluster):
        """
        Find the record that minimizes the information loss of the given cluster.

        Iterates through available records and calculates the potential
        increase in information loss if each record were added to the
        given cluster, then pick the one that causes the lowest loss.

        Parameters
        ----------
        data : list[list]
            The list of available records.
        cluster : list[list]
            The given cluster.

        Returns
        -------
        tuple
            (the best record, its index, the resulting information loss).
        """
        f = partial(self.get_information_loss, cluster=cluster)
        information_losses = (
            self.__parallel.perform(f, data)
            if self.is_parallel
            else [f(record) for record in data]
        )
        best_idx = argmin(information_losses).item()
        return (data[best_idx], best_idx, information_losses[best_idx])

    def find_best_cluster(self, clusters: list, r: list):
        """
        Assign an orphaned record to the most compatible existing cluster.

        Used at the end of the process to assign any remaining records
        to existing clusters while minimizing added information loss.

        Parameters
        ----------
        clusters : list[list]
            The list of already formed clusters.
        r : list
            The record to be assigned.

        Returns
        -------
        tuple
            (index of the best cluster, the resulting information loss).
        """
        information_losses = (
            self.__parallel.perform(
                self.get_information_loss, [r] * len(clusters), clusters
            )
            if self.is_parallel
            else [self.get_information_loss(r, cluster) for cluster in clusters]
        )
        best_idx = argmin(information_losses).item()
        return (best_idx, information_losses[best_idx])

    def do_local_recoding(self):
        """
        Perform the K-Member clustering algorithm.

        The workflow consists of:

        1. Pick a seed record (a random record for the 1st iteration,
           the furthest record from the previous seed otherwise).

        2. Build a cluster of size k by greedily adding records that
           minimize information loss.

        3. Repeat 1-3 until fewer than k records remain.

        4. Distribute remaining records to the most suitable existing
           clusters.

        Returns
        -------
        list
            The final list of clusters.
        """
        data = self.anon_data.values.tolist()

        clusters = []
        information_losses = []
        r_i = None

        if self.is_parallel:
            print(f"Parallelize with {self.cpu_cores} core(s).")
            self.__parallel.activate()

        progress_bar = tqdm(
            total=len(data),
            desc="   Clustering Progress",
            bar_format=_bar_format,
        )

        while len(data) >= self.k:
            if r_i is None:
                random.seed(self.seed)
                r_i_idx = random.randrange(len(data))
                r_i = data[r_i_idx]
            else:
                r_i, r_i_idx = self.find_furthest_record_from_r(r_i, data)
            data.pop(r_i_idx)
            this_cluster = [r_i]
            this_information_loss = None

            while len(this_cluster) < self.k:
                r_j, r_j_idx, this_information_loss = self.find_best_record(
                    data, this_cluster
                )
                data.pop(r_j_idx)
                this_cluster.append(r_j)

            information_losses.append(this_information_loss)
            clusters.append(this_cluster)
            progress_bar.update(self.k)

        for r in data:
            cluster_idx, information_loss = self.find_best_cluster(clusters, r)
            information_losses[cluster_idx] = information_loss
            clusters[cluster_idx].append(r)
            progress_bar.update(1)

        self.information_loss = sum(information_losses)
        progress_bar.close()

        if self.is_parallel:
            self.__parallel.deactivate()

        return clusters
