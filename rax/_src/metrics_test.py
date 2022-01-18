# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pytype: skip-file
"""Tests for rax._src.metrics."""

import doctest
import functools
import math
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import test_util as jtu
import jax.numpy as jnp

import rax
from rax._src import metrics

# Export symbols from math for conciser test value definitions.
log2 = math.log2


class MetricsTest(jtu.JaxTestCase, parameterized.TestCase):

  @parameterized.parameters([{
      "metric_fn": metrics.mrr_metric,
      "expected_value": 1 / 2
  }, {
      "metric_fn": functools.partial(metrics.recall_metric, topn=None),
      "expected_value": 1.
  }, {
      "metric_fn": functools.partial(metrics.recall_metric, topn=3),
      "expected_value": 1.
  }, {
      "metric_fn": functools.partial(metrics.precision_metric, topn=None),
      "expected_value": 0.5
  }, {
      "metric_fn": functools.partial(metrics.precision_metric, topn=3),
      "expected_value": 2. / 3.
  }, {
      "metric_fn": metrics.ap_metric,
      "expected_value": (0.5 + 2. / 3.) / 2.
  }, {
      "metric_fn": metrics.dcg_metric,
      "expected_value": 1 / log2(1 + 2) + 1 / log2(1 + 3)
  }, {
      "metric_fn":
          metrics.ndcg_metric,
      "expected_value": (1 / log2(1 + 2) + 1 / log2(1 + 3)) /
                        (1 / log2(1 + 1) + 1 / log2(1 + 2))
  }])
  def test_computes_metric_value(self, metric_fn, expected_value):
    scores = jnp.asarray([0., 3., 1., 2.])
    labels = jnp.asarray([0., 0., 1., 1.])

    metric = metric_fn(scores, labels)

    self.assertArraysAllClose(jnp.asarray(expected_value), metric)

  @parameterized.parameters([{
      "metric_fn": metrics.mrr_metric,
      "expected_value": [1 / 2, 1 / 3]
  }, {
      "metric_fn": functools.partial(metrics.recall_metric, topn=None),
      "expected_value": [1., 1.]
  }, {
      "metric_fn": functools.partial(metrics.recall_metric, topn=3),
      "expected_value": [1., 1. / 2.]
  }, {
      "metric_fn": functools.partial(metrics.precision_metric, topn=None),
      "expected_value": [0.5, 0.5]
  }, {
      "metric_fn": functools.partial(metrics.precision_metric, topn=3),
      "expected_value": [2. / 3., 1. / 3.]
  }, {
      "metric_fn": metrics.ap_metric,
      "expected_value": [(0.5 + 2. / 3.) / 2., (1. / 3. + 0.5) / 2.]
  }, {
      "metric_fn":
          metrics.dcg_metric,
      "expected_value": [
          1 / log2(1 + 2) + 1 / log2(1 + 3),
          1 / log2(1 + 3) + (2**2 - 1) / log2(1 + 4)
      ]
  }, {
      "metric_fn":
          metrics.ndcg_metric,
      "expected_value": [(1 / log2(1 + 2) + 1 / log2(1 + 3)) /
                         (1 / log2(1 + 1) + 1 / log2(1 + 2)),
                         (1 / log2(1 + 3) + (2**2 - 1) / log2(1 + 4)) /
                         ((2**2 - 1) / log2(1 + 1) + 1 / log2(1 + 2))]
  }])
  def test_computes_metric_value_on_batch_with_vmap(self, metric_fn,
                                                    expected_value):
    scores = jnp.asarray([[0., 3., 1., 2.], [1., 4., 3., 2.]])
    labels = jnp.asarray([[0., 0., 1., 1.], [2., 0., 0., 1.]])
    vmap_metric_fn = jax.vmap(metric_fn, in_axes=(0, 0), out_axes=0)

    metric = vmap_metric_fn(scores, labels)

    self.assertArraysAllClose(jnp.asarray(expected_value), metric)

  @parameterized.parameters([{
      "metric_fn": metrics.mrr_metric,
      "expected_value": [1. / 2., 1.],
      "normalizer": 2.
  }, {
      "metric_fn": functools.partial(metrics.precision_metric, topn=2),
      "expected_value": [1. / 2., 1. / 2.],
      "normalizer": 2.
  }, {
      "metric_fn": functools.partial(metrics.recall_metric, topn=2),
      "expected_value": [1. / 1., 1. / 2.],
      "normalizer": 2.
  }, {
      "metric_fn": metrics.ap_metric,
      "expected_value": [1. / 2., (1. + 2. / 3.) / 2.],
      "normalizer": 2.
  }, {
      "metric_fn": metrics.dcg_metric,
      "expected_value": [1. / log2(1 + 2), 3. / log2(1 + 1) + 1. / log2(1 + 3)],
      "normalizer": 2.
  }, {
      "metric_fn": metrics.ndcg_metric,
      "expected_value": [
          (1. / log2(1 + 2)),
          (3. / log2(1 + 1) + 1. / log2(1 + 3)) / (3. + 1. / log2(1 + 2))
      ],
      "normalizer": 2.
  }])
  def test_computes_reduced_metric(self, metric_fn, expected_value, normalizer):
    scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
    labels = jnp.array([[1., 0., 0.], [0., 1., 2.]])
    expected_value = jnp.asarray(expected_value)

    mean_metric = metric_fn(scores, labels, reduce_fn=jnp.mean)
    sum_metric = metric_fn(scores, labels, reduce_fn=jnp.sum)

    self.assertArraysAllClose(mean_metric, jnp.sum(expected_value) / normalizer)
    self.assertArraysAllClose(sum_metric, jnp.sum(expected_value))

  @parameterized.parameters([(metrics.mrr_metric, (2,))])
  def test_computes_unreduced_metric(self, metric_fn, expected_shape):
    scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
    labels = jnp.array([[1., 0., 0.], [0., 1., 2.]])

    result = metric_fn(scores, labels, reduce_fn=None)

    self.assertEqual(result.shape, expected_shape)

  @parameterized.parameters([{
      "metric_fn": metrics.dcg_metric,
      "expected_value": 2 / log2(1 + 2) + 1 / log2(1 + 3)
  }, {
      "metric_fn":
          metrics.ndcg_metric,
      "expected_value": (2 / log2(1 + 2) + 1 / log2(1 + 3)) /
                        (2 / log2(1 + 1) + 1 / log2(1 + 2))
  }])
  def test_computes_weighted_metric(self, metric_fn, expected_value):
    scores = jnp.asarray([0., 3., 1., 2.])
    labels = jnp.asarray([0., 0., 1., 1.])
    weights = jnp.asarray([1., 2., 1., 2.])

    metric = metric_fn(scores, labels, weights=weights)

    self.assertArraysAllClose(jnp.asarray(expected_value), metric)

  @parameterized.parameters([{
      "metric_fn": metrics.mrr_metric,
      "expected_value": 0.0
  }, {
      "metric_fn": metrics.recall_metric,
      "expected_value": 0.0
  }, {
      "metric_fn": metrics.precision_metric,
      "expected_value": 0.0
  }, {
      "metric_fn": metrics.ap_metric,
      "expected_value": 0.0
  }, {
      "metric_fn": metrics.dcg_metric,
      "expected_value": 0.0
  }, {
      "metric_fn": metrics.ndcg_metric,
      "expected_value": 0.0
  }])
  def test_computes_metric_with_topn_1(self, metric_fn, expected_value):
    scores = jnp.asarray([0., 3., 1., 2.])
    labels = jnp.asarray([1., 0., 1., 1.])

    metric = metric_fn(scores, labels, topn=1)

    self.assertArraysAllClose(jnp.asarray(expected_value), metric)

  @parameterized.parameters([{
      "metric_fn": metrics.mrr_metric,
      "expected_value": 1 / 2
  }, {
      "metric_fn": metrics.recall_metric,
      "expected_value": 2 / 3
  }, {
      "metric_fn": metrics.precision_metric,
      "expected_value": 2 / 3
  }, {
      "metric_fn": metrics.ap_metric,
      "expected_value": (1 / 2 + 2 / 3) / 3
  }, {
      "metric_fn": metrics.dcg_metric,
      "expected_value": 1 / log2(1 + 2) + 1 / log2(1 + 3)
  }, {
      "metric_fn":
          metrics.ndcg_metric,
      "expected_value": (1 / log2(1 + 2) + 1 / log2(1 + 3)) /
                        (1 / log2(1 + 1) + 1 / log2(1 + 2) + 1 / log2(1 + 3))
  }])
  def test_computes_metric_with_topn_3(self, metric_fn, expected_value):
    scores = jnp.asarray([0., 3., 1., 2.])
    labels = jnp.asarray([1., 0., 1., 1.])

    metric = metric_fn(scores, labels, topn=3)

    self.assertArraysAllClose(jnp.asarray(expected_value), metric)

  @parameterized.parameters([{
      "metric_fn": metrics.mrr_metric,
      "expected_value": 0.
  }, {
      "metric_fn": functools.partial(metrics.recall_metric, topn=None),
      "expected_value": 0.
  }, {
      "metric_fn": functools.partial(metrics.recall_metric, topn=3),
      "expected_value": 0.
  }, {
      "metric_fn": functools.partial(metrics.precision_metric, topn=None),
      "expected_value": 0.
  }, {
      "metric_fn": functools.partial(metrics.precision_metric, topn=3),
      "expected_value": 0.
  }, {
      "metric_fn": metrics.ap_metric,
      "expected_value": 0.
  }, {
      "metric_fn": metrics.dcg_metric,
      "expected_value": 0.
  }, {
      "metric_fn": metrics.ndcg_metric,
      "expected_value": 0.
  }])
  def test_computes_metric_value_with_all_masked(self, metric_fn,
                                                 expected_value):
    scores = jnp.asarray([0., 3., 1., 2.])
    labels = jnp.asarray([0., 0., 1., 1.])
    mask = jnp.asarray([False, False, False, False])

    metric = metric_fn(scores, labels, mask=mask)

    self.assertArraysAllClose(jnp.asarray(expected_value), metric)

  @parameterized.parameters([{
      "metric_fn": metrics.mrr_metric,
      "expected_value": 0.
  }, {
      "metric_fn": functools.partial(metrics.recall_metric, topn=None),
      "expected_value": 0.
  }, {
      "metric_fn": functools.partial(metrics.recall_metric, topn=3),
      "expected_value": 0.
  }, {
      "metric_fn": functools.partial(metrics.precision_metric, topn=None),
      "expected_value": 0.
  }, {
      "metric_fn": functools.partial(metrics.precision_metric, topn=3),
      "expected_value": 0.
  }, {
      "metric_fn": metrics.ap_metric,
      "expected_value": 0.
  }, {
      "metric_fn": metrics.dcg_metric,
      "expected_value": 0.
  }, {
      "metric_fn": metrics.ndcg_metric,
      "expected_value": 0.
  }])
  def test_computes_metric_value_with_no_relevant_labels(
      self, metric_fn, expected_value):
    scores = jnp.asarray([0., 3., 1., 2.])
    labels = jnp.asarray([0., 0., 0., 0.])

    metric = metric_fn(scores, labels)

    self.assertArraysAllClose(jnp.asarray(expected_value), metric)

  @parameterized.parameters([{
      "metric_fn": metrics.mrr_metric,
      "expected_value": 1. / 2.
  }, {
      "metric_fn": metrics.precision_metric,
      "expected_value": 1. / 3.
  }, {
      "metric_fn": metrics.recall_metric,
      "expected_value": 1. / 2.
  }, {
      "metric_fn": metrics.ap_metric,
      "expected_value": (1. / 2.) / 2.
  }, {
      "metric_fn": metrics.dcg_metric,
      "expected_value": 1. / log2(1 + 2)
  }, {
      "metric_fn": metrics.ndcg_metric,
      "expected_value": 1. / log2(1 + 2) / (1. / log2(1 + 1) + 1. / log2(1 + 2))
  }])
  def test_treats_neginf_as_unranked_items(self, metric_fn, expected_value):
    scores = jnp.array([-jnp.inf, 5, 2, -jnp.inf, 1])
    labels = jnp.array([1., 0, 1, 0, 0])

    metric = metric_fn(scores, labels)

    self.assertArraysAllClose(jnp.array(expected_value), metric)


class RetrievedItemsTest(jtu.JaxTestCase, parameterized.TestCase):

  def test_does_not_retrieve_items_with_neginf_scores(self):
    scores = jnp.array([-2., -jnp.inf, 4., 3.])
    ranks = jnp.array([3., 4., 1., 2.])

    retrieved_items = metrics._retrieved_items(scores, ranks)

    self.assertArraysEqual(jnp.array([1., 0, 1, 1]), retrieved_items)

  def test_does_not_retrieve_masked_items(self):
    scores = jnp.array([-2., 1., 4., 3.])
    ranks = jnp.array([4., 3., 1., 2.])
    mask = jnp.array([True, False, True, True])

    retrieved_items = metrics._retrieved_items(scores, ranks, mask=mask)

    self.assertArraysEqual(jnp.array([1., 0, 1, 1]), retrieved_items)

  @parameterized.parameters([(0, [0., 0, 0, 0]), (1, [0., 0, 1, 0]),
                             (2, [0., 0, 1, 1]), (3, [0., 1, 1, 1]),
                             (4, [1., 1, 1, 1]), (10, [1., 1, 1, 1]),
                             (None, [1., 1, 1, 1])])
  def test_does_not_retrieve_items_beyond_topn(self, topn, expected):
    scores = jnp.array([-2., 1., 4., 3.])
    ranks = jnp.array([4., 3., 1., 2.])

    retrieved_items = metrics._retrieved_items(scores, ranks, topn=topn)

    self.assertArraysEqual(jnp.array(expected), retrieved_items)


def load_tests(loader, tests, ignore):
  del loader, ignore  # Unused.
  tests.addTests(
      doctest.DocTestSuite(
          metrics,
          globs={
              "functools": functools,
              "jax": jax,
              "jnp": jnp,
              "rax": rax
          }))
  return tests


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
