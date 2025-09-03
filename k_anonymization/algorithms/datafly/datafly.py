# +
from numpy import argmax

from ..utils import generalize, is_k_anonymized
from ..type import Algorithm


# -

class Datafly(Algorithm):
    # No suppression at the moment
    def pick_attribute(self, df, qids):
        # pick_attribute_with_highest_cardinality
        cardinalities = [len(df[x].unique().tolist()) for x in qids]
        return qids[argmax(cardinalities)]

    def anonymize(self):
        qids = [self.org_data.columns[x] for x in self.dataset.props["qi_index"]]
        hierarchies_tracking = {}

        while True:
            if is_k_anonymized(self.anon_data, self.k, qids):
                break
            else:
                generalized_att = self.pick_attribute(self.anon_data, qids)
                if generalized_att in list(hierarchies_tracking):
                    hierarchies_tracking[generalized_att] = (
                        hierarchies_tracking[generalized_att] + 1
                    )
                else:
                    hierarchies_tracking[generalized_att] = 0
                generalize(
                    self.anon_data,
                    self.dataset.hierarchies[generalized_att],
                    hierarchies_tracking[generalized_att],
                )
        return self.anon_data
