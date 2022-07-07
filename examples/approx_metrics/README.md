# Approximate Metric Optimization

This example demonstrates how to use the Rax transformations to build
approximate metric losses and use them to optimize a linear model on the WEB10K
dataset.

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
python examples/approx_metrics/main.py
```

You should see the following expected output in the form of a JSON dictionary
containing the different approximate metrics and their results.

```json
{
  "ApproxAP": {
    "AP": 0.5947682857513428,
    "NDCG": 0.6587949991226196,
    "R@50": 0.5807018876075745
  },
  "ApproxNDCG": {
    "AP": 0.587692141532898,
    "NDCG": 0.6700138449668884,
    "R@50": 0.5744442939758301
  },
  "ApproxR@50": {
    "AP": 0.5849018096923828,
    "NDCG": 0.6449851989746094,
    "R@50": 0.5746314525604248
  }
}
```
