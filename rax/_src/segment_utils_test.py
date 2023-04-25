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
"""Tests for rax._src.segment_utils."""

import doctest
import functools
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

import rax
from rax._src import segment_utils


class SameSegmentMaskTest(absltest.TestCase):

  def test_returns_pairwise_mask_indicating_same_segments(self):
    segments = jnp.asarray([0, 0, 1])
    expected = jnp.asarray([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    actual = jnp.int32(segment_utils.same_segment_mask(segments))
    np.testing.assert_array_equal(actual, expected)


class SegmentSumTest(absltest.TestCase):

  def test_segment_sum(self):
    scores = jnp.asarray([1.0, 2.0, 4.0])
    segments = jnp.asarray([0, 0, 1])
    expected = jnp.asarray([3.0, 3.0, 4.0])
    actual = segment_utils.segment_sum(scores, segments)
    np.testing.assert_array_equal(actual, expected)


class SegmentMaxTest(absltest.TestCase):

  def test_computes_max_per_segment(self):
    scores = jnp.array([1.0, 2.0, 4.0, -5.0, -5.5, -4.5])
    segments = jnp.array([0, 0, 1, 2, 2, 2])
    expected = jnp.array([2.0, 2.0, 4.0, -4.5, -4.5, -4.5])
    actual = segment_utils.segment_max(scores, segments)
    np.testing.assert_array_equal(actual, expected)

  def test_computes_max_with_initial_value(self):
    scores = jnp.array([1.0, 2.0, 4.0, 1.0, 2.0, 3.0])
    segments = jnp.array([0, 0, 1, 2, 2, 2])
    expected = jnp.array([2.5, 2.5, 4.0, 3.0, 3.0, 3.0])
    actual = segment_utils.segment_max(scores, segments, initial=2.5)
    np.testing.assert_array_equal(actual, expected)

  def test_computes_max_with_mask(self):
    scores = jnp.array([1.0, 2.0, 4.0, 1.0, 2.0, 3.0])
    segments = jnp.array([0, 0, 1, 2, 2, 2])
    mask = jnp.array([1, 0, 1, 1, 1, 0])
    actual = segment_utils.segment_max(scores, segments, where=mask)
    # Only non-masked entries have well-defined behavior under max, so we only
    # check those.
    np.testing.assert_equal(actual[0], jnp.array(1.0))
    np.testing.assert_equal(actual[2], jnp.array(4.0))
    np.testing.assert_equal(actual[3], jnp.array(2.0))
    np.testing.assert_equal(actual[4], jnp.array(2.0))


class SegmentLogSoftmaxTest(absltest.TestCase):

  def test_computes_log_softmax(self):
    a = jnp.array([2.0, -1.0, 0.9, 1.3, 2.3])
    segments = jnp.array([0, 0, 0, 1, 1])

    expected_1 = jax.nn.log_softmax(a[0:3])
    expected_2 = jax.nn.log_softmax(a[3:5])
    actual = segment_utils.segment_log_softmax(a, segments)

    np.testing.assert_allclose(actual[0:3], expected_1)
    np.testing.assert_allclose(actual[3:5], expected_2)

  def test_computes_log_softmax_with_mask(self):
    a = jnp.array([2.0, -1.0, 0.9, 1.3, 2.3])
    segments = jnp.array([0, 0, 0, 1, 1])
    mask = jnp.array([1, 1, 0, 1, 1])

    expected_1 = jax.nn.log_softmax(a[0:3], where=mask[0:3], initial=jnp.min(a))
    expected_2 = jax.nn.log_softmax(a[3:5], where=mask[3:5], initial=jnp.min(a))
    actual = segment_utils.segment_log_softmax(a, segments, where=mask)

    np.testing.assert_allclose(actual[0:2], expected_1[0:2])
    np.testing.assert_allclose(actual[3:5], expected_2)

  def test_handles_extreme_values(self):
    a = jnp.array([-1e34, 3e28, 0.00000001, -100001, 0.999])
    segments = jnp.array([1, 1, 1, 9999, 9999])

    expected_1 = jax.nn.log_softmax(a[0:3])
    expected_2 = jax.nn.log_softmax(a[3:5])
    actual = segment_utils.segment_log_softmax(a, segments)

    np.testing.assert_allclose(actual[0:3], expected_1)
    np.testing.assert_allclose(actual[3:5], expected_2)


class InSegmentIndicesTest(absltest.TestCase):

  def test_in_segment_indices(self):
    segments = jnp.asarray([0, 0, 0, 1, 2, 2])
    expected = jnp.asarray([0, 1, 2, 0, 0, 1])
    actual = segment_utils.in_segment_indices(segments)
    np.testing.assert_array_equal(actual, expected)

  def test_in_segment_indices_unordered(self):
    segments = jnp.asarray([0, 0, 1, 0, 2, 2])
    expected = jnp.asarray([0, 1, 0, 2, 0, 1])
    actual = segment_utils.in_segment_indices(segments)
    np.testing.assert_array_equal(actual, expected)


class FirstItemSegmentMask(absltest.TestCase):

  def test_selects_first_item_per_segment(self):
    segments = jnp.array([0, 0, 1, 1, 1, 2, 2, 1, 1, 3, 3, 3])
    expected = jnp.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=jnp.bool_)
    actual = segment_utils.first_item_segment_mask(segments)
    np.testing.assert_array_equal(actual, expected)

  def test_does_not_select_masked_items(self):
    segments = jnp.array([0, 0, 1, 1, 1, 2, 2, 1, 1, 3, 3, 3])
    where = jnp.array([1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0], dtype=jnp.bool_)
    expected = jnp.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0], dtype=jnp.bool_)
    actual = segment_utils.first_item_segment_mask(segments, where=where)
    np.testing.assert_array_equal(actual, expected)


def load_tests(loader, tests, ignore):
  del loader, ignore  # Unused.
  tests.addTests(
      doctest.DocTestSuite(
          segment_utils,
          globs={
              "functools": functools,
              "jax": jax,
              "jnp": jnp,
              "rax": rax,
              "segment_utils": segment_utils,
          },
      )
  )
  return tests


if __name__ == "__main__":
  absltest.main()
