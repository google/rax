# Flax Integration

This example demonstrates how to use Flax to build and train a neural ranking
model that is optimized with a Rax loss and evaluated with Rax metrics.

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
python examples/flax_integration/main.py
```

You should see the following expected output in the form of a JSON dictionary
containing the per-epoch loss and ranking metrics.

```json
[
  {
    "epoch": 1,
    "loss": 371.3304748535156,
    "metric/mrr": 0.8062829971313477,
    "metric/ndcg": 0.6677320003509521,
    "metric/ndcg@10": 0.4055347740650177
  },
  {
    "epoch": 2,
    "loss": 370.42974853515625,
    "metric/mrr": 0.8242350220680237,
    "metric/ndcg": 0.6812514662742615,
    "metric/ndcg@10": 0.43049752712249756
  },
  {
    "epoch": 3,
    "loss": 370.25244140625,
    "metric/mrr": 0.8261540532112122,
    "metric/ndcg": 0.6834192276000977,
    "metric/ndcg@10": 0.4342570900917053
  }
]
```
