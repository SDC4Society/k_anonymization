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

# Create a Dataset

+++

## `Dataset` instance

+++

A `Dataset` instance stores original data and its important properties, such as names of the quasi-identifier (QID) attributes, their generalization hierarchies, etc.

Let's load an example dataset instance and call the method `describe()` to check its properties.

```{code-cell} ipython3
from k_anonymization.datasets import load_mini_crime

dataset = load_mini_crime()
dataset.describe()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

As shown, this dataset has 3 QID attributes. 'marital-status' is a categorical attribute. 'age' and 'zip' are numerical attributes. 'crime' is the target attribute (for data analysis).

To access the generalization hierarchy of a QID attribute, e.g., 'zip', simply call `dataset.hierarchies['zip']`.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-input]
---
dataset.hierarchies['zip']
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Create a `Dataset` instance

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### From in-memory data

Consider the following data:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
import pandas as pd

df = pd.DataFrame({
    "age": [22, 27, 26, 31, 33, 38],
    "sex": ["Male", "Male", "Male", "Female", "Female", "Male"],
    "hometown": ["Ibaraki", "Tokyo", "Ibaraki", "Iwate", "Iwate", "Aomori"],
    "condition": ["Fever", "Healthy", "Fat", "Dizzy", "Nausea", "Sleepy"],
})
df
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The simpliest way to create a `Dataset` instance is:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from k_anonymization.core import Dataset

dataset = Dataset(name="Test Dataset", df=df)
dataset.describe()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

As shown, simply calling `Dataset(#NAME, #DF)` automatically treats all attributes as QID attributes. Whether a QID attribute is numerical or categorical is also determined automatically from data (`df.dtypes`).

The target attribute can be specified by the argument `target`. If only `target` is provided, the `Dataset` instance automatically treats all attributes except `target` as QID attributes.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
dataset = Dataset(name="Test Dataset", df=df, target="condition")
dataset.info
```

To manually determine whether a QID attribute is numerical or categorical, pass a list of booleans to the argument `is_categorical` (`False` if numerical, `True` if categorical). Note that the length of `is_categorical` must match the length of QID attributes, and the order of values in `is_categorical` must match the order of `df.columns`.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-output]
---
dataset = Dataset(
    name="Test Dataset", 
    df=df, 
    target="condition",
    is_categorical=[False, True, True],
                 #["age", "sex", "hometown"]
                 #[ Num., Cat., Cat.]
)
dataset.info
```

To manually select QID attribute, pass a list of QID attribute names to the argument `qids`. Note that the order of values in `qids` must match the order of `df.columns`. If `is_categorical` is also provided, the length of `is_categorical` must match the length of `qids`, as well as the orders of their values.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-output]
---
dataset = Dataset(
    name="Test Dataset", 
    df=df, 
    target="condition",
    qids=["age", "hometown"],
    is_categorical=[False, False],
)
dataset.info
```

Generalization hierarchies for QID attributes can be by passing a dictionary `{#ATTR_NAME: #DF}` the argument `hierarchies`. If any attribute is not provided in `hierarchies`, it will be assigned a default hierarchy, i.e., all values are suppressed. If `hierarchies` is not provided, all attributes are assigned default hierarchies.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from k_anonymization.core import HierarchiesDict

dataset = Dataset(
    name="Test Dataset", 
    df=df, 
    target="condition",
    hierarchies=HierarchiesDict({
        "hometown": pd.DataFrame([
            ["Ibaraki", "Kanto" , "*"],
            ["Tokyo"  , "Kanto" , "*"],
            ["Iwate"  , "Tohoku", "*"],
            ["Aomori" , "Tohoku", "*"],
        ])
    }, df)
)
dataset.hierarchies["hometown"]
```

:::{seealso}
[](create-hierarchy)
:::

### From directory

+++ {"editable": true, "slideshow": {"slide_type": ""}}

A `Dataset` instance can be created from data and properties files in a directory by calling `Dataset.from_dir(#NAME, #PATH_TO_DIR)`. 
The directory contains:

- `data.csv` (Required): a CSV file storing data, where the top
  row stores the header (attribute) names.

- `props.json` (Optional): a JSON file storing data properties,
  including `qids`, `is_categorical`, and `target`. If any of those
  properties is not provided, its default value as when calling `Dataset()`
  is used. If `props.json` does not exist, default values are used
  for all properties.

- `hierarchies/` (Optional): a directory containing hierarchy files
  of all QID attributes. If the hierarchy file of an attribute is not
  provided, its default hierarchy is used. If `hierarchies/` does not
  exist, default hierarchies are used for all attributes.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
dataset = Dataset.from_dir("Test Dataset", "./test_dataset")
dataset.describe()
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---

```
