---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# Basic Usage

+++

First, let's load an example dataset.

```{code-cell} ipython3
from k_anonymization.datasets import load_mini_crime

dataset = load_mini_crime()
dataset
```

To anonymize the above dataset, create an `Algorithm` instance, then call the `anonymize()` method.

In general, any `Algorithm` instance requires a dataset instance containing original data and its properties (in this case, variable `dataset` in the above code) and the `k`-anonymity value.

```{code-cell} ipython3
from k_anonymization.algorithms.full_generalization import Datafly

algorithm = Datafly(dataset, k=2)
algorithm.anonymize()
algorithm.anon_data
```

```{code-cell} ipython3
from k_anonymization.algorithms.local_recoding import KMember

algorithm = KMember(dataset, k=2, seed=2026)
algorithm.anonymize()
algorithm.anon_data
```

```{code-cell} ipython3

```
