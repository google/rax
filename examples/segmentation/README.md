# Segmentation

This example demonstrates how to optimize a neural ranking model using a
segmented data format.

Segmented data (sometimes referred to as "packing") refers to flat data arrays
where a specific segments array indicates which elements should be grouped
together to form lists. This data format is more compact than a padded approach.
Rax losses and metrics support using segmented data via the `segments` keyword
argument.

## Instructions

Clone the Rax repo:

```shell
git clone git@github.com:google/rax.git
cd rax
```

Install the example dependencies:

```shell
pip install -r requirements/requirements-examples.txt
```

And then run the example code:

```shell
python examples/segmentation/main.py
```

You should see the following expected output in the form of a JSON dictionary
containing the loss and ranking metrics at different steps during training.

```json
[
  {
    "step": 200,
    "loss": 340.78642868041993,
    "mrr": 0.887844353467226,
    "ndcg": 0.7206529039144516,
    "ndcg@10": 0.48478462554514407
  },
  {
    "step": 400,
    "loss": 333.58397720336916,
    "mrr": 0.9479761835932732,
    "ndcg": 0.8064527478814125,
    "ndcg@10": 0.6433215755224228
  },
  {
    "step": 600,
    "loss": 329.4338551330566,
    "mrr": 0.9637777781486512,
    "ndcg": 0.8601924273371696,
    "ndcg@10": 0.7409686753153801
  }
]
```
