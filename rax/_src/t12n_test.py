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
"""Tests for rax._src.t12n."""

import doctest
import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import test_util as jtu
import jax.numpy as jnp

import rax
from rax._src import metrics
from rax._src import t12n


class ApproxT12nTest(jtu.JaxTestCase, parameterized.TestCase):

  @parameterized.parameters([
      metrics.mrr_metric,
      functools.partial(metrics.recall_metric, topn=2),
      functools.partial(metrics.precision_metric, topn=2),
      metrics.ap_metric,
      metrics.dcg_metric,
      metrics.ndcg_metric,
  ])
  def test_approx_t12n_metric_has_nonzero_nonnan_loss(self, metric_fn):
    scores = jnp.asarray([-2., 1., 3., 9.])
    labels = jnp.asarray([1., 0., 1., 0.])

    loss_fn = t12n.approx_t12n(metric_fn)
    loss = loss_fn(scores, labels)

    self.assertArraysEqual(jnp.isnan(loss), jnp.zeros_like(jnp.isnan(loss)))
    self.assertArraysEqual(loss != 0., jnp.ones_like(loss, dtype=jnp.bool_))

  @parameterized.parameters([
      metrics.mrr_metric,
      functools.partial(metrics.recall_metric, topn=2),
      functools.partial(metrics.precision_metric, topn=2),
      metrics.ap_metric,
      metrics.dcg_metric,
      metrics.ndcg_metric,
  ])
  def test_approx_t12n_metric_has_nonzero_nonnan_grads(self, metric_fn):
    scores = jnp.asarray([-2., 1., 3., 9.])
    labels = jnp.asarray([1., 0., 1., 0.])

    loss_fn = t12n.approx_t12n(metric_fn)
    grads = jax.grad(loss_fn)(scores, labels)

    self.assertArraysEqual(jnp.isnan(grads), jnp.zeros_like(jnp.isnan(grads)))
    self.assertArraysEqual(grads != 0., jnp.ones_like(grads, dtype=jnp.bool_))

  @parameterized.parameters([
      metrics.mrr_metric,
      functools.partial(metrics.recall_metric, topn=2),
      functools.partial(metrics.precision_metric, topn=2),
      metrics.ap_metric,
      metrics.dcg_metric,
      metrics.ndcg_metric,
  ])
  def test_approx_t12n_metric_has_nonnan_grads_with_all_mask(self, metric_fn):
    scores = jnp.asarray([-2., 1., 3., 9.])
    labels = jnp.asarray([1., 0., 1., 0.])
    mask = jnp.asarray([False, False, False, False])

    loss_fn = t12n.approx_t12n(metric_fn)
    grads = jax.grad(loss_fn)(scores, labels, mask=mask)

    self.assertArraysEqual(jnp.isnan(grads), jnp.zeros_like(jnp.isnan(grads)))

  @parameterized.parameters([
      metrics.mrr_metric,
      functools.partial(metrics.recall_metric, topn=2),
      functools.partial(metrics.precision_metric, topn=2),
      metrics.ap_metric,
      metrics.dcg_metric,
      metrics.ndcg_metric,
  ])
  def test_approx_t12n_metric_has_nonnan_grads_with_zero_labels(
      self, metric_fn):
    scores = jnp.asarray([-2., 1., 3., 9.])
    labels = jnp.asarray([0., 0., 0., 0.])

    loss_fn = t12n.approx_t12n(metric_fn)
    grads = jax.grad(loss_fn)(scores, labels)

    self.assertArraysEqual(jnp.isnan(grads), jnp.zeros_like(jnp.isnan(grads)))


class BoundT12nTest(jtu.JaxTestCase, parameterized.TestCase):

  def test_computes_upper_bound_on_ranks(self):
    scores = jnp.array([2., -1.5, 0.9])
    labels = jnp.ones_like(scores)

    def fn(scores, labels, *, rank_fn):
      del labels  # unused.
      return -rank_fn(scores)

    bound_fn = t12n.bound_t12n(fn)
    ranks = bound_fn(scores, labels)

    expected = jnp.array([(1. + 0. + 0.), (1. + 4.5 + 3.4), (1. + 2.1 + 0.)])
    self.assertArraysAllClose(ranks, expected)

  def test_computes_lower_bound_on_cutoffs(self):
    scores = jnp.array([2., -1.5, 0.9])
    labels = jnp.ones_like(scores)

    def fn(scores, labels, *, cutoff_fn):
      del labels  # unused.
      return -cutoff_fn(scores, n=2)

    bound_fn = t12n.bound_t12n(fn)
    ranks = bound_fn(scores, labels)

    expected = jnp.array([1., -1.5 - 0.9, 0.])
    self.assertArraysAllClose(ranks, expected)

  @parameterized.parameters([
      metrics.mrr_metric,
      functools.partial(metrics.recall_metric, topn=2),
      functools.partial(metrics.precision_metric, topn=2),
      metrics.ap_metric,
      metrics.dcg_metric,
      metrics.ndcg_metric,
  ])
  def test_bound_t12n_metric_has_nonzero_nonnan_loss(self, metric_fn):
    scores = jnp.asarray([-2., 1., 3., 9.])
    labels = jnp.asarray([1., 0., 1., 0.])

    loss_fn = t12n.bound_t12n(metric_fn)
    loss = loss_fn(scores, labels)

    self.assertArraysEqual(jnp.isnan(loss), jnp.zeros_like(jnp.isnan(loss)))
    self.assertArraysEqual(loss != 0., jnp.ones_like(loss, dtype=jnp.bool_))

  @parameterized.parameters([
      metrics.mrr_metric,
      functools.partial(metrics.recall_metric, topn=2),
      functools.partial(metrics.precision_metric, topn=2),
      metrics.ap_metric,
      metrics.dcg_metric,
      metrics.ndcg_metric,
  ])
  def test_bound_t12n_metric_has_nonzero_nonnan_grads(self, metric_fn):
    scores = jnp.asarray([-2., 1., 3., 9.])
    labels = jnp.asarray([1., 0., 1., 0.])

    loss_fn = t12n.bound_t12n(metric_fn)
    grads = jax.grad(loss_fn)(scores, labels)

    self.assertArraysEqual(jnp.isnan(grads), jnp.zeros_like(jnp.isnan(grads)))
    self.assertGreater(jnp.sum(jnp.abs(grads)), 0.)

  @parameterized.parameters([
      metrics.mrr_metric,
      functools.partial(metrics.recall_metric, topn=2),
      functools.partial(metrics.precision_metric, topn=2),
      metrics.ap_metric,
      metrics.dcg_metric,
      metrics.ndcg_metric,
  ])
  def test_bound_t12n_metric_has_nonnan_grads_with_all_mask(self, metric_fn):
    scores = jnp.asarray([-2., 1., 3., 9.])
    labels = jnp.asarray([1., 0., 1., 0.])
    mask = jnp.asarray([False, False, False, False])

    loss_fn = t12n.bound_t12n(metric_fn)
    grads = jax.grad(loss_fn)(scores, labels, mask=mask)

    self.assertArraysEqual(jnp.isnan(grads), jnp.zeros_like(jnp.isnan(grads)))

  @parameterized.parameters([
      metrics.mrr_metric,
      functools.partial(metrics.recall_metric, topn=2),
      functools.partial(metrics.precision_metric, topn=2),
      metrics.ap_metric,
      metrics.dcg_metric,
      metrics.ndcg_metric,
  ])
  def test_bound_t12n_metric_has_nonnan_grads_with_zero_labels(self, metric_fn):
    scores = jnp.asarray([-2., 1., 3., 9.])
    labels = jnp.asarray([0., 0., 0., 0.])

    loss_fn = t12n.bound_t12n(metric_fn)
    grads = jax.grad(loss_fn)(scores, labels)

    self.assertArraysEqual(jnp.isnan(grads), jnp.zeros_like(jnp.isnan(grads)))


class GumbelT12nTest(jtu.JaxTestCase):

  def test_samples_scores_using_key(self):
    scores = jnp.asarray([0., 1., 2.])
    labels = jnp.asarray([0., 1., 0.])
    mock_loss_fn = lambda scores, labels: scores

    new_loss_fn = t12n.gumbel_t12n(mock_loss_fn, samples=1)

    loss = new_loss_fn(scores, labels, gumbel_key=jax.random.PRNGKey(42))
    self.assertArraysAllClose(loss, jnp.asarray([[0.589013, 0.166654,
                                                  0.962401]]))

  def test_repeats_inputs_n_times(self):
    scores = jnp.asarray([0., 1., 2.])
    labels = jnp.asarray([0., 1., 0.])
    mask = jnp.asarray([True, True, False])
    n = 32
    mock_loss_fn = lambda scores, labels, mask: (scores, labels, mask)

    new_loss_fn = t12n.gumbel_t12n(mock_loss_fn, samples=n)

    new_scores, new_labels, new_mask = new_loss_fn(
        scores, labels, mask=mask, gumbel_key=jax.random.PRNGKey(42))
    self.assertEqual(new_scores.shape, (n, 3))
    self.assertEqual(new_labels.shape, (n, 3))
    self.assertEqual(new_mask.shape, (n, 3))

  def test_samples_scores_using_gumbel_beta_shape(self):
    scores = jnp.asarray([0., 1., 2.])
    labels = jnp.asarray([0., 1., 0.])
    mock_loss_fn = lambda scores, labels: scores

    new_loss_fn = t12n.gumbel_t12n(mock_loss_fn, samples=1)

    loss = new_loss_fn(
        scores, labels, gumbel_key=jax.random.PRNGKey(42), gumbel_beta=0.00001)
    self.assertArraysAllClose(loss, jnp.expand_dims(scores, 0), atol=1e-3)

  def test_handles_extreme_scores(self):
    scores = jnp.asarray([-3e18, 1., 2e22])
    labels = jnp.asarray([0., 1., 0.])
    mock_loss_fn = lambda scores, labels: scores

    new_loss_fn = t12n.gumbel_t12n(mock_loss_fn, samples=1)

    loss = new_loss_fn(scores, labels, gumbel_key=jax.random.PRNGKey(42))
    self.assertArraysAllClose(loss, jnp.asarray([[-3e18, 1.666543e-01, 2e22]]))

  def test_raises_an_error_if_no_gumbel_key_is_provided(self):
    scores = jnp.asarray([-3e18, 1., 2e22])
    labels = jnp.asarray([0., 1., 0.])
    mock_loss_fn = lambda scores, labels: scores

    new_loss_fn = t12n.gumbel_t12n(mock_loss_fn)

    with self.assertRaises(TypeError):
      new_loss_fn(scores, labels)


def load_tests(loader, tests, ignore):
  del loader, ignore  # Unused.
  tests.addTests(
      doctest.DocTestSuite(t12n, globs={
          "jax": jax,
          "jnp": jnp,
          "rax": rax
      }))
  return tests


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
