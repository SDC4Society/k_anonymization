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

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Create Generalization Hierarchies

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Some algorithms requires generalization hierarchies of the QID attributes.
A generalization hierarchy is represented as a `pandas.DataFrame` (called a _hierarchy dataframe_), whose column names are progressive integers denoting the levels of generalization. Column `0` stores the unique original values of the attribute in the data.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from k_anonymization.core import Dataset

dataset = Dataset.from_dir("Test Dataset", "./test_dataset")
dataset.hierarchies["hometown"]
```

## Create a `Hierarchy` instance

+++

A `Hierarchy` instance can be created from a hierarchy dataframe.

```{code-cell} ipython3
import pandas as pd
from k_anonymization.core import Hierarchy

Hierarchy(
    name="hometown", 
    hierarchy_df=pd.DataFrame([
        ["Ibaraki", "Kanto" , "*"],
        ["Tokyo"  , "Kanto" , "*"],
        ["Iwate"  , "Tohoku", "*"],
        ["Aomori" , "Tohoku", "*"],
    ])
)
```

For attributes with high cardinality (typically numerical attributes), creating a hierarchy dataframe with all the unique values can be tough. Therefore, `Hierarchy` can also be created from a dictionary storing the lambda functions to compute generalized values. The format of the dictionary is as follows:

```python
{
  "lambda":[
      "lambda x: 'x0 to x1'", # logic to generalize original values to level 1 
      "lambda x: 'x1 to x2'", # logic to generalize values at level 1 to level 2
      "lambda x: 'x2 to x3'", # logic to generalize values at level 2 to level 3
      ...
      "lambda x: '*'", # suppress values (the highest level)
  ]
}
```

```{code-cell} ipython3
props = {
  "lambda": [
    "lambda x: f'{x//5*5}~{x//5*5+4}'",
    "lambda x: f'{int(x.split(\"~\")[0])//10*10}~{int(x.split(\"~\")[0])//10*10+9}'",
    "lambda x: '*'"
  ]
}

Hierarchy.from_dict(name="age", org_column=dataset.df["age"], props=props)
```

The `Hierarchy` class also allows loading hierarchy dataframe from a CSV file via method `Hierarchy.from_csv()`, or loading lambda function definitions from a JSON file via `Hierarchy.from_json()`

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-output]
---
Hierarchy.from_csv(name="hometown", path="./test_dataset/hierarchies/hometown.csv")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-output]
---
Hierarchy.from_json(
    name="age", 
    org_column=dataset.df["age"],
    path="./test_dataset/hierarchies/age.json"
)
```

To create a default hierarchy (i.e., all values are suppressed), use `Hierarchy.make_default()`

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-output]
---
Hierarchy.make_default(name="age", org_column=dataset.df["age"])
```

## Create a `HierarchiesDict`

+++ {"editable": true, "slideshow": {"slide_type": ""}}

In a `Dataset` instance, a `HierarchiesDict` is used to manage the `Hierarchy` instances.
`HierarchiesDict` allows getting an attribute's hierarchy both by its name and its column index.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from k_anonymization.core import HierarchiesDict

hierarchies = HierarchiesDict(
    hierarchies_dict={
        "hometown": pd.DataFrame([
            ["Ibaraki", "Kanto" , "*"],
            ["Tokyo"  , "Kanto" , "*"],
            ["Iwate"  , "Tohoku", "*"],
            ["Aomori" , "Tohoku", "*"],
        ])
    },
    df=dataset.df # Provide `df` to match column indices.
    # If `df` is not provided, default indices follow
    # `hierarchies_dict.keys()`
)
hierarchies["hometown"] # or hierarchies[2]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

When passing a `HierarchiesDict` to a `Dataset` initialization, any missing `Hierarchy` instance in `HierarchiesDict` will be automatically set to a default hierarchy.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
df = pd.DataFrame({
    "age": [22, 27, 26, 31, 33, 38],
    "sex": ["Male", "Male", "Male", "Female", "Female", "Male"],
    "hometown": ["Ibaraki", "Tokyo", "Ibaraki", "Iwate", "Iwate", "Aomori"],
    "condition": ["Fever", "Healthy", "Fat", "Dizzy", "Nausea", "Sleepy"],
})

hierarchies = HierarchiesDict(
    hierarchies_dict={
        "hometown": pd.DataFrame([
            ["Ibaraki", "Kanto" , "*"],
            ["Tokyo"  , "Kanto" , "*"],
            ["Iwate"  , "Tohoku", "*"],
            ["Aomori" , "Tohoku", "*"],
        ])
    },
    df=df
)

dataset = Dataset("Test Dataset", df, hierarchies=hierarchies)
dataset.hierarchies["age"] # or hierarchies[0]
```

To create default hierarchies for all QID attributes, use `HierarchiesDict.make_default()`

```{code-cell} ipython3
hierarchies = HierarchiesDict.make_default(df, qids=["age", "sex", "hometown"])
hierarchies["hometown"]
```
