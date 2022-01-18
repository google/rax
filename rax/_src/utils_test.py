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
"""Tests for rax._src.utils."""

import doctest
from absl.testing import absltest
import jax
from jax import test_util as jtu
import jax.numpy as jnp

import rax
from rax._src import utils


class NormalizeProbabilitiesTest(jtu.JaxTestCase):

  def test_sums_to_one_for_given_axis(self):
    arr = jnp.asarray([[0., 1., 2.], [3., 4., 5.]])

    result1 = utils.normalize_probabilities(arr, axis=0)
    result2 = utils.normalize_probabilities(arr, axis=1)

    self.assertArraysEqual(
        result1, jnp.asarray([[0., 1. / 5., 2. / 7.], [1., 4. / 5., 5. / 7.]]))
    self.assertArraysEqual(
        result2,
        jnp.asarray([[0., 1. / 3., 2. / 3.], [3. / 12., 4. / 12., 5. / 12.]]))

  def test_sums_to_one_for_default_axis(self):
    arr = jnp.asarray([[0., 1., 2.], [3., 4., 5.]])

    result = utils.normalize_probabilities(arr)

    self.assertArraysEqual(
        result,
        jnp.asarray([[0., 1. / 3., 2. / 3.], [3. / 12., 4. / 12., 5. / 12.]]))

  def test_handles_mask(self):
    arr = jnp.asarray([[0., 1., 2.], [3., 4., 5.]])
    mask = jnp.asarray([[True, False, True], [True, True, True]])

    result = utils.normalize_probabilities(arr, mask, axis=1)

    self.assertArraysEqual(
        jnp.sum(result, axis=1, where=mask), jnp.asarray([1., 1.]))

  def test_correctly_sets_all_zeros(self):
    arr = jnp.asarray([[0., 0., 0.], [0., 0., 0.]])

    result1 = utils.normalize_probabilities(arr, axis=0)
    result2 = utils.normalize_probabilities(arr, axis=1)

    self.assertArraysEqual(jnp.sum(result1, axis=0), jnp.asarray([1., 1., 1.]))
    self.assertArraysEqual(jnp.sum(result2, axis=1), jnp.asarray([1., 1.]))

  def test_correctly_handles_all_masked(self):
    arr = jnp.asarray([[2., 1., 3.], [1., 1., 1.]])
    mask = jnp.asarray([[False, False, False], [False, False, False]])

    result1 = utils.normalize_probabilities(arr, mask, axis=0)
    result2 = utils.normalize_probabilities(arr, mask, axis=1)

    self.assertArraysEqual(jnp.sum(result1, axis=0), jnp.asarray([1., 1., 1.]))
    self.assertArraysEqual(jnp.sum(result2, axis=1), jnp.asarray([1., 1.]))


class SortByTest(jtu.JaxTestCase):

  def test_sorts_by_scores(self):
    scores = jnp.asarray([0., 3., 1., 2.])
    tensors_to_sort = [jnp.asarray([10., 13., 11., 12.])]

    result = utils.sort_by(scores, tensors_to_sort)[0]

    self.assertArraysEqual(result, jnp.asarray([13., 12., 11., 10.]))

  def test_sorts_by_given_axis(self):
    scores = jnp.asarray([[3., 1., 2.], [1., 5., 3.]])
    tensors_to_sort = [jnp.asarray([[0., 1., 2.], [3., 4., 5.]])]

    result_0 = utils.sort_by(scores, tensors_to_sort, axis=0)[0]
    result_1 = utils.sort_by(scores, tensors_to_sort, axis=1)[0]

    self.assertArraysEqual(result_0, jnp.asarray([[0., 4., 5.], [3., 1., 2.]]))
    self.assertArraysEqual(result_1, jnp.asarray([[0., 2., 1.], [4., 5., 3.]]))

  def test_sorts_multiple_tensors(self):
    scores = jnp.asarray([0., 3., 1., 2.])
    tensors_to_sort = [
        jnp.asarray([10., 13., 11., 12.]),
        jnp.asarray([50., 56., 52., 54.]),
        jnp.asarray([75., 78., 76., 77.])
    ]

    result = utils.sort_by(scores, tensors_to_sort)

    self.assertArraysEqual(result[0], jnp.asarray([13., 12., 11., 10.]))
    self.assertArraysEqual(result[1], jnp.asarray([56., 54., 52., 50.]))
    self.assertArraysEqual(result[2], jnp.asarray([78., 77., 76., 75.]))

  def test_places_masked_values_last(self):
    scores = jnp.asarray([0., 3., 1., 2.])
    tensors_to_sort = [jnp.asarray([10., 13., 11., 12.])]
    mask = jnp.asarray([True, True, False, False])

    result = utils.sort_by(scores, tensors_to_sort, mask=mask)[0]

    self.assertArraysEqual(result, jnp.asarray([13., 10., 12., 11.]))

  def test_breaks_ties_randomly_when_rng_key_is_provided(self):
    scores = jnp.asarray([0., 1., 1., 2.])
    tensors_to_sort = [jnp.asarray([10., 11.1, 11.2, 12.])]
    rng_key = jax.random.PRNGKey(4242)
    rng_key1, rng_key2 = jax.random.split(rng_key)

    result1 = utils.sort_by(scores, tensors_to_sort, rng_key=rng_key1)[0]
    result2 = utils.sort_by(scores, tensors_to_sort, rng_key=rng_key2)[0]

    self.assertArraysEqual(result1, jnp.asarray([12., 11.2, 11.1, 10.]))
    self.assertArraysEqual(result2, jnp.asarray([12., 11.1, 11.2, 10.]))


class SortRankTest(jtu.JaxTestCase):

  def test_ranks_by_sorting_scores(self):
    scores = jnp.asarray([[0., 1., 2.], [2., 1., 3.]])

    ranks = utils.sort_ranks(scores)

    self.assertArraysEqual(ranks, jnp.asarray([[3, 2, 1], [2, 3, 1]]))

  def test_ranks_along_given_axis(self):
    scores = jnp.asarray([[0., 1., 2.], [1., 2., 0.]])

    ranks = utils.sort_ranks(scores, axis=0)

    self.assertArraysEqual(ranks, jnp.asarray([[2, 2, 1], [1, 1, 2]]))

  def test_ranks_with_ties_broken_randomly(self):
    scores = jnp.asarray([2., 1., 1.])
    key = jax.random.PRNGKey(1)
    key1, key2 = jax.random.split(key)

    ranks1 = utils.sort_ranks(scores, rng_key=key1)
    ranks2 = utils.sort_ranks(scores, rng_key=key2)

    self.assertArraysEqual(ranks1, jnp.asarray([1, 2, 3]))
    self.assertArraysEqual(ranks2, jnp.asarray([1, 3, 2]))


class ApproxRanksTest(jtu.JaxTestCase):

  def test_computes_approx_ranks(self):
    scores = jnp.asarray([-3., 1., 2.])

    ranks = utils.approx_ranks(scores)

    sigmoid = jax.nn.sigmoid
    self.assertArraysEqual(
        ranks,
        jnp.asarray([
            sigmoid(3. + 1.) + sigmoid(3. + 2.) + 1.0,
            sigmoid(-1. - 3.) + sigmoid(-1. + 2.) + 1.0,
            sigmoid(-2. - 3.) + sigmoid(-2. + 1.) + 1.0
        ]))

  def test_maintains_order(self):
    scores = jnp.asarray([-4., 1., -3., 2.])

    ranks = utils.approx_ranks(scores)
    true_ranks = utils.sort_ranks(scores)

    self.assertArraysEqual(jnp.argsort(ranks), jnp.argsort(true_ranks))

  def test_computes_approx_ranks_with_mask(self):
    scores_without_mask = jnp.asarray([3.33, 1.125])
    scores = jnp.asarray([3.33, 2.5, 1.125])
    mask = jnp.asarray([True, False, True])

    ranks = utils.approx_ranks(scores_without_mask)
    ranks_with_mask = utils.approx_ranks(scores, mask=mask)

    self.assertArraysEqual(
        ranks, jnp.asarray([ranks_with_mask[0], ranks_with_mask[2]]))


def load_tests(loader, tests, ignore):
  del loader, ignore  # Unused.
  tests.addTests(
      doctest.DocTestSuite(
          utils, extraglobs={
              "jax": jax,
              "jnp": jnp,
              "rax": rax
          }))
  return tests


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
