# 🦖 **Rax**: Composable Learning to Rank using JAX

[![Docs](https://readthedocs.org/projects/rax/badge/?version=latest)](https://rax.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/rax?color=brightgreen)](https://pypi.org/project/rax/)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/google/rax/blob/main/LICENSE)

**Rax** is a Learning-to-Rank library written in JAX. Rax provides off-the-shelf
implementations of ranking losses and metrics to be used with JAX. It provides
the following functionality:

- Ranking losses (`rax.*_loss`): `rax.softmax_loss`,
  `rax.pairwise_logistic_loss`, ...
- Ranking metrics (`rax.*_metric`): `rax.mrr_metric`, `rax.ndcg_metric`, ...
- Transformations (`rax.*_t12n`): `rax.approx_t12n`, `rax.gumbel_t12n`, ...

## Ranking

A ranking problem is different from traditional classification/regression
problems in that its objective is to optimize for the correctness of the
**relative order** of a **list of examples** (e.g., documents) for a given
context (e.g., a query). **Rax** provides support for ranking problems within
the JAX ecosystem. It can be used in, but is not limited to, the following
applications:

- **Search**: ranking a list of documents with respect to a query.
- **Recommendation**: ranking a list of items given a user as context.
- **Question Answering**: finding the best answer from a list of candidates.
- **Dialogue System**: finding the best response from a list of responses.

## Synopsis

In a nutshell, given the scores and labels for a list of items, Rax can compute
various ranking losses and metrics:

```python
import jax.numpy as jnp
import rax

scores = jnp.asarray([2.2, -1.3, 5.4])  # output of a model.
labels = jnp.asarray([1., 0., 0.])      # indicates doc 1 is relevant.

rax.ndcg_metric(scores, labels)         # computes a ranking metric.
rax.pairwise_hinge_loss(scores, labels) # computes a ranking loss.
```

All of the Rax losses and metrics are purely functional and compose well with
standard JAX transformations. Additionally, Rax provides ranking-specific
transformations so you can build new ranking losses. An example is
`rax.approx_t12n`, which can be used to transform any (non-differentiable)
ranking metric into a differentiable loss. For example:

```python
loss_fn = rax.approx_t12n(rax.ndcg_metric)
loss_fn(scores, labels)            # differentiable approx ndcg loss.
jax.grad(loss_fn)(scores, labels)  # computes gradients w.r.t. scores.
```

## Installation

See https://github.com/google/jax#installation for instructions on installing JAX.

We suggest installing the latest stable version of Rax by running:

`$ pip install rax`

## Examples

See the `examples/` directory for complete examples on how to use Rax.

## Citing Rax

If you use Rax, please consider citing our
[paper](https://research.google/pubs/pub51453/):

```
@inproceedings{jagerman2022rax,
  title = {Rax: Composable Learning-to-Rank using JAX},
  author  = {Rolf Jagerman and Xuanhui Wang and Honglei Zhuang and Zhen Qin and
  Michael Bendersky and Marc Najork},
  year  = {2022},
  booktitle = {Proceedings of the 28th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining}
}
```
