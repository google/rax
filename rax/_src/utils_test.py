# Copyright 2023 Google LLC.
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
"""Tests for rax._src.utils."""

import doctest
import inspect
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

import rax
from rax._src import utils


class NormalizeProbabilitiesTest(absltest.TestCase):

  def test_sums_to_one_for_given_axis(self):
    arr = jnp.asarray([[0., 1., 2.], [3., 4., 5.]])

    result1 = utils.normalize_probabilities(arr, axis=0)
    result2 = utils.normalize_probabilities(arr, axis=1)

    np.testing.assert_array_equal(
        result1, jnp.asarray([[0., 1. / 5., 2. / 7.], [1., 4. / 5., 5. / 7.]]))
    np.testing.assert_array_equal(
        result2,
        jnp.asarray([[0., 1. / 3., 2. / 3.], [3. / 12., 4. / 12., 5. / 12.]]))

  def test_sums_to_one_for_default_axis(self):
    arr = jnp.asarray([[0., 1., 2.], [3., 4., 5.]])

    result = utils.normalize_probabilities(arr)

    np.testing.assert_array_equal(
        result,
        jnp.asarray([[0., 1. / 3., 2. / 3.], [3. / 12., 4. / 12., 5. / 12.]]))

  def test_handles_where(self):
    arr = jnp.asarray([[0., 1., 2.], [3., 4., 5.]])
    where = jnp.asarray([[True, False, True], [True, True, True]])

    result = utils.normalize_probabilities(arr, where, axis=1)

    np.testing.assert_array_equal(
        jnp.sum(result, axis=1, where=where), jnp.asarray([1., 1.]))

  def test_correctly_sets_all_zeros(self):
    arr = jnp.asarray([[0., 0., 0.], [0., 0., 0.]])

    result1 = utils.normalize_probabilities(arr, axis=0)
    result2 = utils.normalize_probabilities(arr, axis=1)

    np.testing.assert_array_equal(
        jnp.sum(result1, axis=0), jnp.asarray([1., 1., 1.]))
    np.testing.assert_array_equal(
        jnp.sum(result2, axis=1), jnp.asarray([1., 1.]))

  def test_correctly_handles_all_masked(self):
    arr = jnp.asarray([[2., 1., 3.], [1., 1., 1.]])
    where = jnp.asarray([[False, False, False], [False, False, False]])

    result1 = utils.normalize_probabilities(arr, where, axis=0)
    result2 = utils.normalize_probabilities(arr, where, axis=1)

    np.testing.assert_array_equal(
        jnp.sum(result1, axis=0), jnp.asarray([1., 1., 1.]))
    np.testing.assert_array_equal(
        jnp.sum(result2, axis=1), jnp.asarray([1., 1.]))


class LogCumsumExp(absltest.TestCase):

  def test_computes_logcumsumexp(self):
    x = jnp.asarray([-4., 5., 2.3, 0.])

    result = utils.logcumsumexp(x)

    np.testing.assert_array_equal(
        result,
        jnp.asarray([
            jnp.log(jnp.exp(-4.)),
            jnp.log(jnp.exp(-4.) + jnp.exp(5.)),
            jnp.log(jnp.exp(-4.) + jnp.exp(5.) + jnp.exp(2.3)),
            jnp.log(jnp.exp(-4.) + jnp.exp(5.) + jnp.exp(2.3) + jnp.exp(0.))
        ]))

  def test_computes_over_specified_axis(self):
    x = jnp.asarray([[-4., 2.3, 0.], [2.2, -1.2, 1.1]])

    result = utils.logcumsumexp(x, axis=-1)
    np.testing.assert_array_equal(result[0, :], utils.logcumsumexp(x[0, :]))
    np.testing.assert_array_equal(result[1, :], utils.logcumsumexp(x[1, :]))

    result = utils.logcumsumexp(x, axis=0)
    np.testing.assert_array_equal(result[:, 0], utils.logcumsumexp(x[:, 0]))
    np.testing.assert_array_equal(result[:, 1], utils.logcumsumexp(x[:, 1]))
    np.testing.assert_array_equal(result[:, 2], utils.logcumsumexp(x[:, 2]))

  def test_computes_reversed(self):
    x = jnp.asarray([-4., 5., 2.3, 0.])
    x_flipped = jnp.asarray([0., 2.3, 5., -4.])

    result_reverse = utils.logcumsumexp(x, reverse=True)
    result_flipped = jnp.flip(utils.logcumsumexp(x_flipped))

    np.testing.assert_array_equal(result_reverse, result_flipped)

  def test_computes_with_where_mask(self):
    x = jnp.asarray([-4., 5., 2.3, 0.])
    where = jnp.asarray([True, False, True, True])
    x_masked = jnp.asarray([-4., 2.3, 0.])

    result_where = utils.logcumsumexp(x, where=where)
    result_masked = utils.logcumsumexp(x_masked)

    np.testing.assert_array_equal(result_where[0], result_masked[0])
    np.testing.assert_array_equal(result_where[2], result_masked[1])
    np.testing.assert_array_equal(result_where[3], result_masked[2])

  def test_handles_extreme_values(self):
    x = jnp.asarray([-4., -2.1e26, 5., 3.4e38, 10., -2.99e26])

    result = utils.logcumsumexp(x)

    np.testing.assert_array_equal(
        result, jnp.asarray([-4., -4., 5.0001235, 3.4e38, 3.4e38, 3.4e38]))


class SortByTest(absltest.TestCase):

  def test_sorts_by_scores(self):
    scores = jnp.asarray([0., 3., 1., 2.])
    tensors_to_sort = [jnp.asarray([10., 13., 11., 12.])]

    result = utils.sort_by(scores, tensors_to_sort)[0]

    np.testing.assert_array_equal(result, jnp.asarray([13., 12., 11., 10.]))

  def test_sorts_by_given_axis(self):
    scores = jnp.asarray([[3., 1., 2.], [1., 5., 3.]])
    tensors_to_sort = [jnp.asarray([[0., 1., 2.], [3., 4., 5.]])]

    result_0 = utils.sort_by(scores, tensors_to_sort, axis=0)[0]
    result_1 = utils.sort_by(scores, tensors_to_sort, axis=1)[0]

    np.testing.assert_array_equal(result_0,
                                  jnp.asarray([[0., 4., 5.], [3., 1., 2.]]))
    np.testing.assert_array_equal(result_1,
                                  jnp.asarray([[0., 2., 1.], [4., 5., 3.]]))

  def test_sorts_multiple_tensors(self):
    scores = jnp.asarray([0., 3., 1., 2.])
    tensors_to_sort = [
        jnp.asarray([10., 13., 11., 12.]),
        jnp.asarray([50., 56., 52., 54.]),
        jnp.asarray([75., 78., 76., 77.])
    ]

    result = utils.sort_by(scores, tensors_to_sort)

    np.testing.assert_array_equal(result[0], jnp.asarray([13., 12., 11., 10.]))
    np.testing.assert_array_equal(result[1], jnp.asarray([56., 54., 52., 50.]))
    np.testing.assert_array_equal(result[2], jnp.asarray([78., 77., 76., 75.]))

  def test_places_masked_values_last(self):
    scores = jnp.asarray([0., 3., 1., 2.])
    tensors_to_sort = [jnp.asarray([10., 13., 11., 12.])]
    where = jnp.asarray([True, True, False, False])

    result = utils.sort_by(scores, tensors_to_sort, where=where)[0]

    np.testing.assert_array_equal(result, jnp.asarray([13., 10., 12., 11.]))

  def test_breaks_ties_randomly_when_key_is_provided(self):
    scores = jnp.asarray([0., 1., 1., 2.])
    tensors_to_sort = [jnp.asarray([10., 11.1, 11.2, 12.])]
    key = jax.random.PRNGKey(4242)
    key1, key2 = jax.random.split(key)

    result1 = utils.sort_by(scores, tensors_to_sort, key=key1)[0]
    result2 = utils.sort_by(scores, tensors_to_sort, key=key2)[0]

    np.testing.assert_array_equal(result1, jnp.asarray([12., 11.2, 11.1, 10.]))
    np.testing.assert_array_equal(result2, jnp.asarray([12., 11.1, 11.2, 10.]))


class CutoffTest(absltest.TestCase):

  def test_cutoff(self):
    scores = jnp.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 3.0]])
    results = utils.cutoff(scores, n=2)
    np.testing.assert_array_equal(results, jnp.asarray([[0, 1, 1], [1, 0, 1]]))

  def test_cutoff_with_where(self):
    scores = jnp.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 3.0]])
    where = jnp.asarray([[False, False, True], [True, True, True]])
    results = utils.cutoff(scores, n=2, where=where)
    np.testing.assert_array_equal(results, jnp.asarray([[0, 0, 1], [1, 0, 1]]))

  def test_cutoff_with_segments(self):
    scores = jnp.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 3.0]])
    segments = jnp.asarray([[0, 0, 1], [0, 1, 1]])
    results = utils.cutoff(scores, n=1, segments=segments)
    np.testing.assert_array_equal(results, jnp.asarray([[0, 1, 1], [1, 0, 1]]))

  def test_cutoff_with_segments_and_where(self):
    scores = jnp.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 3.0]])
    segments = jnp.asarray([[0, 1, 1], [0, 1, 1]])
    where = jnp.asarray([[False, True, False], [True, True, True]])
    results = utils.cutoff(scores, n=2, segments=segments, where=where)
    np.testing.assert_array_equal(results, jnp.asarray([[0, 1, 0], [1, 1, 1]]))

  def test_cutoff_no_n(self):
    scores = jnp.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 0.0]])
    results = utils.cutoff(scores, n=None)
    np.testing.assert_array_equal(results, jnp.asarray([[1, 1, 1], [1, 1, 1]]))

  def test_cutoff_large_n(self):
    scores = jnp.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 0.0]])
    results = utils.cutoff(scores, n=4)
    np.testing.assert_array_equal(results, jnp.asarray([[1, 1, 1], [1, 1, 1]]))

  def test_cutoff_n_is_zero(self):
    scores = jnp.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 0.0]])
    results = utils.cutoff(scores, n=0)
    np.testing.assert_array_equal(results, jnp.asarray([[0, 0, 0], [0, 0, 0]]))


class ApproxCutoffTest(absltest.TestCase):

  def test_approx_cutoff(self):
    scores = jnp.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 3.0]])
    results = utils.approx_cutoff(scores, n=1)
    np.testing.assert_allclose(
        results,
        jnp.asarray(
            [[0.182426, 0.377541, 0.622459], [0.377541, 0.182426, 0.622459]]
        ),
        rtol=1e-5,
    )

  def test_approx_cutoff_hard(self):
    scores = jnp.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 3.0]])
    results = utils.approx_cutoff(scores, n=2, step_fn=lambda x: x >= 0)
    np.testing.assert_array_equal(results, jnp.asarray([[0, 1, 1], [1, 0, 1]]))

  def test_approx_cutoff_with_where(self):
    scores = jnp.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 3.0]])
    where = jnp.asarray([[False, False, True], [True, True, True]])
    results = utils.approx_cutoff(
        scores, n=2, where=where, step_fn=lambda x: x >= 0
    )
    np.testing.assert_array_equal(results, jnp.asarray([[0, 0, 1], [1, 0, 1]]))

  def test_approx_cutoff_with_segments(self):
    scores = jnp.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 3.0]])
    segments = jnp.asarray([[0, 0, 1], [0, 1, 1]])
    results = utils.approx_cutoff(
        scores, n=1, segments=segments, step_fn=lambda x: x >= 0
    )
    np.testing.assert_array_equal(results, jnp.asarray([[0, 1, 1], [1, 0, 1]]))

  def test_approx_cutoff_with_segments_and_where(self):
    scores = jnp.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 3.0]])
    segments = jnp.asarray([[0, 1, 1], [0, 1, 1]])
    where = jnp.asarray([[False, True, False], [True, True, True]])
    results = utils.approx_cutoff(
        scores, n=2, segments=segments, where=where, step_fn=lambda x: x >= 0
    )
    np.testing.assert_array_equal(results, jnp.asarray([[0, 1, 0], [1, 1, 1]]))

  def test_approx_cutoff_no_n(self):
    scores = jnp.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 0.0]])
    results = utils.approx_cutoff(scores, n=None, step_fn=lambda x: x >= 0)
    np.testing.assert_array_equal(results, jnp.asarray([[1, 1, 1], [1, 1, 1]]))

  def test_approx_cutoff_large_n(self):
    scores = jnp.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 0.0]])
    results = utils.approx_cutoff(scores, n=4, step_fn=lambda x: x >= 0)
    np.testing.assert_array_equal(results, jnp.asarray([[1, 1, 1], [1, 1, 1]]))

  def test_approx_cutoff_n_is_zero(self):
    scores = jnp.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 0.0]])
    results = utils.approx_cutoff(scores, n=0, step_fn=lambda x: x >= 0)
    np.testing.assert_array_equal(results, jnp.asarray([[0, 0, 0], [0, 0, 0]]))


class RanksTest(absltest.TestCase):

  def test_ranks_by_sorting_scores(self):
    scores = jnp.asarray([[0., 1., 2.], [2., 1., 3.]])

    ranks = utils.ranks(scores)

    np.testing.assert_array_equal(ranks, jnp.asarray([[3, 2, 1], [2, 3, 1]]))

  def test_ranks_along_given_axis(self):
    scores = jnp.asarray([[0., 1., 2.], [1., 2., 0.]])

    ranks = utils.ranks(scores, axis=0)

    np.testing.assert_array_equal(ranks, jnp.asarray([[2, 2, 1], [1, 1, 2]]))

  def test_ranks_with_ties_broken_randomly(self):
    scores = jnp.asarray([2., 1., 1.])
    key = jax.random.PRNGKey(1)
    key1, key2 = jax.random.split(key)

    ranks1 = utils.ranks(scores, key=key1)
    ranks2 = utils.ranks(scores, key=key2)

    np.testing.assert_array_equal(ranks1, jnp.asarray([1, 2, 3]))
    np.testing.assert_array_equal(ranks2, jnp.asarray([1, 3, 2]))

  def test_ranks_with_segments(self):
    scores = jnp.asarray([1.0, 3.0, 2.0, 1.0, 2.0, 3.0])
    segments = jnp.asarray([0, 0, 0, 1, 2, 2])
    expected = jnp.asarray([3, 1, 2, 1, 2, 1])
    ranks = utils.ranks(scores, segments=segments)
    np.testing.assert_array_equal(ranks, expected)

  def test_ranks_with_segments_and_where(self):
    scores = jnp.asarray([1.0, 3.0, 2.0, 1.0, 2.0, 3.0])
    segments = jnp.asarray([0, 0, 0, 1, 2, 2])
    where = jnp.asarray([False, True, True, True, True, False])
    expected = jnp.asarray([3, 1, 2, 1, 1, 2])
    ranks = utils.ranks(scores, segments=segments, where=where)
    np.testing.assert_array_equal(ranks, expected)


class ApproxRanksTest(absltest.TestCase):

  def test_computes_approx_ranks(self):
    scores = jnp.asarray([-3., 1., 2.])

    ranks = utils.approx_ranks(scores)

    sigmoid = jax.nn.sigmoid
    np.testing.assert_array_equal(
        ranks,
        jnp.asarray([
            sigmoid(3. + 1.) + sigmoid(3. + 2.) + 1.0,
            sigmoid(-1. - 3.) + sigmoid(-1. + 2.) + 1.0,
            sigmoid(-2. - 3.) + sigmoid(-2. + 1.) + 1.0
        ]))

  def test_maintains_order(self):
    scores = jnp.asarray([-4., 1., -3., 2.])

    ranks = utils.approx_ranks(scores)
    true_ranks = utils.ranks(scores)

    np.testing.assert_array_equal(jnp.argsort(ranks), jnp.argsort(true_ranks))

  def test_computes_approx_ranks_with_where(self):
    scores_without_where = jnp.asarray([3.33, 1.125])
    scores = jnp.asarray([3.33, 2.5, 1.125])
    where = jnp.asarray([True, False, True])

    ranks = utils.approx_ranks(scores_without_where)
    ranks_with_where = utils.approx_ranks(scores, where=where)

    np.testing.assert_array_equal(
        ranks, jnp.asarray([ranks_with_where[0], ranks_with_where[2]]))

  def test_computes_approx_ranks_with_segments(self):
    scores_segment_0 = jnp.asarray([3.33, 1.125])
    scores = jnp.asarray([3.33, 2.5, 1.125])
    segments = jnp.asarray([0, 1, 0])

    ranks = utils.approx_ranks(scores_segment_0)
    ranks_with_segments = utils.approx_ranks(scores, segments=segments)

    np.testing.assert_array_equal(
        ranks, jnp.asarray([ranks_with_segments[0], ranks_with_segments[2]])
    )


class SafeReduceTest(absltest.TestCase):

  def test_reduces_values_according_to_fn(self):
    a = jnp.array([[3., 2.], [4.5, 1.2]])

    res_mean = utils.safe_reduce(a, reduce_fn=jnp.mean)
    res_sum = utils.safe_reduce(a, reduce_fn=jnp.sum)
    res_none = utils.safe_reduce(a, reduce_fn=None)

    np.testing.assert_allclose(res_mean, jnp.mean(a))
    np.testing.assert_allclose(res_sum, jnp.sum(a))
    np.testing.assert_allclose(res_none, a)

  def test_reduces_values_with_mask(self):
    a = jnp.array([[3., 2., 0.01], [4.5, 1.2, 0.9]])
    where = jnp.array([[True, False, True], [True, True, False]])

    res_mean = utils.safe_reduce(a, where=where, reduce_fn=jnp.mean)
    res_sum = utils.safe_reduce(a, where=where, reduce_fn=jnp.sum)
    res_none = utils.safe_reduce(a, where=where, reduce_fn=None)

    np.testing.assert_allclose(res_mean, jnp.mean(a, where=where))
    np.testing.assert_allclose(res_sum, jnp.sum(a, where=where))
    np.testing.assert_allclose(res_none, jnp.where(where, a, 0.))

  def test_reduces_mean_with_all_masked(self):
    a = jnp.array([[3., 2., 0.01], [4.5, 1.2, 0.9]])
    where = jnp.array([[False, False, False], [False, False, False]])

    res_mean = utils.safe_reduce(a, where=where, reduce_fn=jnp.mean)

    np.testing.assert_allclose(res_mean, jnp.array(0.))


class ComputePairsTest(absltest.TestCase):

  def test_computes_all_pairs(self):
    a = jnp.array([1., 2., 3.])
    expected = jnp.array([11.0, 21.0, 31.0, 12.0, 22.0, 32.0, 13.0, 23.0, 33.0])

    result = utils.compute_pairs(a, lambda a, b: a + b * 10.0)

    np.testing.assert_allclose(result, expected)

  def test_computes_all_pairs_on_empty_array(self):
    a = jnp.array([])
    expected = jnp.array([])

    result = utils.compute_pairs(a, lambda a, b: a + b * 10.0)

    np.testing.assert_allclose(result, expected)

  def test_computes_all_pairs_with_batch_dimension(self):
    a = jnp.array([[1., 2.], [3., 4.]])
    expected = jnp.array([[1.0, 2.0, 2.0, 4.0], [9.0, 12.0, 12.0, 16.0]])

    result = utils.compute_pairs(a, lambda a, b: a * b)

    np.testing.assert_allclose(result, expected)


class UpdateSignatureTest(absltest.TestCase):

  def test_adds_parameter_to_wrapped_fn(self):
    fn = lambda a, b: a * b

    @utils.update_signature(fn, "key")
    def wrapper(*args, key=None, **kwargs):
      del key  # Unused.
      return fn(*args, **kwargs)

    # The expected signature should be `fn` with `key=None` added to it.
    expected = inspect.signature(lambda a, b, *, key=None: a * b)
    self.assertEqual(inspect.signature(wrapper), expected)

  def test_adds_parameter_to_wrapped_fn_2(self):
    def fn(a, /, b, *, c: int = 0, d: float = 1.0):
      return (a + b + c) * d

    @utils.update_signature(fn, "e")
    def wrapper(*args, e=2.0, **kwargs):
      return e + fn(*args, **kwargs)

    # Expected signature should be the signature of `fn` with an extra kwarg
    # `e=2.0` added to it, like this:
    def expected_fn(a, /, b, *, c: int = 0, d: float = 1.0, e=2.0):
      return e + fn(a, b, c=c, d=d)

    # Check that signatures match.
    signature = inspect.signature(wrapper)
    expected_signature = inspect.signature(expected_fn)
    self.assertEqual(signature, expected_signature)

  def test_does_not_add_parameters_that_already_exist(self):
    def fn(a, b, c):
      del a, b, c  # Unused.

    # This should only add `d` (and not `c` since it already exist).
    @utils.update_signature(fn, "c", "d")
    def wrapper(a, b, c, d=None):
      del d  # Unused.
      return fn(a, b, c)

    def expected(a, b, c, d=None):
      del a, b, c, d

    self.assertEqual(inspect.signature(wrapper), inspect.signature(expected))

  def test_adds_multiple_parameters(self):
    def fn(a, b):
      del a, b  # Unused.

    @utils.update_signature(fn, "c", "d")
    def fn2(*args, c=1, d=5, **kwargs):
      del c, d  # Unused.
      return fn(*args, **kwargs)

    def expected_fn(a, b, *, c=1, d=5):
      del a, b, c, d  # Unused.

    self.assertEqual(inspect.signature(fn2), inspect.signature(expected_fn))

  def test_keeps_signature_when_extra_parameters_is_empty(self):
    def fn(a, b):
      del a, b  # Unused.
      pass

    expected = inspect.signature(fn)

    # No parameters are provided to be updated. Signature should remain the
    # same.
    @utils.update_signature(fn)
    def new_fn(a, b):
      return fn(a, b)

    actual = inspect.signature(new_fn)

    self.assertEqual(actual, expected)


def load_tests(loader, tests, ignore):
  del loader, ignore  # Unused.
  tests.addTests(
      doctest.DocTestSuite(
          utils, extraglobs={
              "jax": jax,
              "jnp": jnp,
              "rax": rax,
              "utils": utils,
          }))
  return tests


if __name__ == "__main__":
  absltest.main()
