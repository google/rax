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
import jax.numpy as jnp
import numpy as np

import rax
from rax._src import metrics
from rax._src import t12n


class ApproxT12nTest(parameterized.TestCase):

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

    np.testing.assert_array_equal(
        jnp.isnan(loss), jnp.zeros_like(jnp.isnan(loss)))
    np.testing.assert_array_equal(loss != 0.,
                                  jnp.ones_like(loss, dtype=jnp.bool_))

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

    np.testing.assert_array_equal(
        jnp.isnan(grads), jnp.zeros_like(jnp.isnan(grads)))
    np.testing.assert_array_equal(grads != 0.,
                                  jnp.ones_like(grads, dtype=jnp.bool_))

  @parameterized.parameters([
      metrics.mrr_metric,
      functools.partial(metrics.recall_metric, topn=2),
      functools.partial(metrics.precision_metric, topn=2),
      metrics.ap_metric,
      metrics.dcg_metric,
      metrics.ndcg_metric,
  ])
  def test_approx_t12n_metric_has_nonnan_grads_with_all_where(self, metric_fn):
    scores = jnp.asarray([-2., 1., 3., 9.])
    labels = jnp.asarray([1., 0., 1., 0.])
    where = jnp.asarray([False, False, False, False])

    loss_fn = t12n.approx_t12n(metric_fn)
    grads = jax.grad(loss_fn)(scores, labels, where=where)

    np.testing.assert_array_equal(
        jnp.isnan(grads), jnp.zeros_like(jnp.isnan(grads)))

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

    np.testing.assert_array_equal(
        jnp.isnan(grads), jnp.zeros_like(jnp.isnan(grads)))


class BoundT12nTest(parameterized.TestCase):

  def test_computes_upper_bound_on_ranks(self):
    scores = jnp.array([2., -1.5, 0.9])
    labels = jnp.ones_like(scores)

    def fn(scores, labels, *, rank_fn):
      del labels  # unused.
      return -rank_fn(scores)

    bound_fn = t12n.bound_t12n(fn)
    ranks = bound_fn(scores, labels)

    expected = jnp.array([(1. + 0. + 0.), (1. + 4.5 + 3.4), (1. + 2.1 + 0.)])
    np.testing.assert_allclose(ranks, expected)

  def test_computes_lower_bound_on_cutoffs(self):
    scores = jnp.array([2., -1.5, 0.9])
    labels = jnp.ones_like(scores)

    def fn(scores, labels, *, cutoff_fn):
      del labels  # unused.
      return -cutoff_fn(scores, n=2)

    bound_fn = t12n.bound_t12n(fn)
    ranks = bound_fn(scores, labels)

    expected = jnp.array([1., -1.5 - (-1.5 + 0.9) / 2., 1.])
    np.testing.assert_allclose(ranks, expected)

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

    np.testing.assert_array_equal(
        jnp.isnan(loss), jnp.zeros_like(jnp.isnan(loss)))
    np.testing.assert_array_equal(loss != 0.,
                                  jnp.ones_like(loss, dtype=jnp.bool_))

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

    np.testing.assert_array_equal(
        jnp.isnan(grads), jnp.zeros_like(jnp.isnan(grads)))
    self.assertGreater(jnp.sum(jnp.abs(grads)), 0.)

  @parameterized.parameters([
      metrics.mrr_metric,
      functools.partial(metrics.recall_metric, topn=2),
      functools.partial(metrics.precision_metric, topn=2),
      metrics.ap_metric,
      metrics.dcg_metric,
      metrics.ndcg_metric,
  ])
  def test_bound_t12n_metric_has_nonnan_grads_with_all_where(self, metric_fn):
    scores = jnp.asarray([-2., 1., 3., 9.])
    labels = jnp.asarray([1., 0., 1., 0.])
    where = jnp.asarray([False, False, False, False])

    loss_fn = t12n.bound_t12n(metric_fn)
    grads = jax.grad(loss_fn)(scores, labels, where=where)

    np.testing.assert_array_equal(
        jnp.isnan(grads), jnp.zeros_like(jnp.isnan(grads)))

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

    np.testing.assert_array_equal(
        jnp.isnan(grads), jnp.zeros_like(jnp.isnan(grads)))


class GumbelT12nTest(parameterized.TestCase):

  def test_samples_scores_using_key(self):
    scores = jnp.asarray([0., 1., 2.])
    labels = jnp.asarray([0., 1., 0.])
    mock_loss_fn = lambda scores, labels: scores

    new_loss_fn = t12n.gumbel_t12n(mock_loss_fn, samples=1)

    loss = new_loss_fn(scores, labels, key=jax.random.PRNGKey(42))
    np.testing.assert_allclose(
        loss, jnp.asarray([[0.589013, 0.166654, 0.962401]]), rtol=1E-5)

  def test_repeats_inputs_n_times(self):
    scores = jnp.asarray([0., 1., 2.])
    labels = jnp.asarray([0., 1., 0.])
    where = jnp.asarray([True, True, False])
    n = 32
    mock_loss_fn = lambda scores, labels, where: (scores, labels, where)

    new_loss_fn = t12n.gumbel_t12n(mock_loss_fn, samples=n)

    new_scores, new_labels, new_where = new_loss_fn(
        scores, labels, where=where, key=jax.random.PRNGKey(42))
    self.assertEqual(new_scores.shape, (n, 3))
    self.assertEqual(new_labels.shape, (n, 3))
    self.assertEqual(new_where.shape, (n, 3))

  def test_samples_scores_using_gumbel_beta_shape(self):
    scores = jnp.asarray([0., 1., 2.])
    labels = jnp.asarray([0., 1., 0.])
    mock_loss_fn = lambda scores, labels: scores

    new_loss_fn = t12n.gumbel_t12n(mock_loss_fn, samples=1, beta=0.00001)

    loss = new_loss_fn(scores, labels, key=jax.random.PRNGKey(42))
    np.testing.assert_allclose(loss, jnp.expand_dims(scores, 0), atol=1e-3)

  def test_handles_extreme_scores(self):
    scores = jnp.asarray([-3e18, 1., 2e22])
    labels = jnp.asarray([0., 1., 0.])
    mock_loss_fn = lambda scores, labels: scores

    new_loss_fn = t12n.gumbel_t12n(mock_loss_fn, samples=1)

    loss = new_loss_fn(scores, labels, key=jax.random.PRNGKey(42))
    np.testing.assert_allclose(
        loss, jnp.asarray([[-3e18, 1.666543e-01, 2e22]]), rtol=1E-5)

  def test_raises_an_error_if_no_key_is_provided(self):
    scores = jnp.asarray([-3e18, 1., 2e22])
    labels = jnp.asarray([0., 1., 0.])
    mock_loss_fn = lambda scores, labels: scores

    new_loss_fn = t12n.gumbel_t12n(mock_loss_fn)

    with self.assertRaises(TypeError):
      new_loss_fn(scores, labels)

  def test_applies_log_softmax_transformation(self):
    scores = jnp.asarray([3., -2., 5.5, 1.])
    labels = jnp.asarray([0., 1., 2., 0.])
    mock_loss_fn = lambda scores, labels: scores

    gumbel_loss_fn = t12n.gumbel_t12n(mock_loss_fn)
    logsoftmax_loss_fn = t12n.gumbel_t12n(mock_loss_fn, smoothing_factor=1e-20)

    output_scores = gumbel_loss_fn(scores, labels, key=jax.random.PRNGKey(42))
    logsoftmax_scores = logsoftmax_loss_fn(
        scores, labels, key=jax.random.PRNGKey(42))

    np.testing.assert_allclose(
        jnp.log(jax.nn.softmax(output_scores) + 1e-20), logsoftmax_scores)


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
  absltest.main()
