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
"""Tests for rax._src.weights."""

import doctest
import math

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

import rax
from rax._src import lambdaweights
from rax._src import losses

discount = lambda rank: 1.0 / math.log2(rank + 1.0)
gain = lambda label: ((2.0**label) - 1.0)


class LambdaweightsTest(parameterized.TestCase):

  @parameterized.parameters([{
      "lambdaweight_fn": lambdaweights.labeldiff_lambdaweight,
      "expected": [0.0, 1.0, 0.3, 1.0, 0.0, 0.7, 0.3, 0.7, 0.0]
  }, {
      "lambdaweight_fn":
          lambdaweights.dcg_lambdaweight,
      "expected": [
          0.0,
          3.0 * abs(discount(2) - discount(3)) * abs(gain(0.0) - gain(1.0)),
          3.0 * abs(discount(1) - discount(3)) * abs(gain(0.0) - gain(0.3)),
          3.0 * abs(discount(3) - discount(2)) * abs(gain(1.0) - gain(0.0)),
          0.0,
          3.0 * abs(discount(1) - discount(2)) * abs(gain(1.0) - gain(0.3)),
          3.0 * abs(discount(3) - discount(1)) * abs(gain(0.3) - gain(0.0)),
          3.0 * abs(discount(2) - discount(1)) * abs(gain(0.3) - gain(1.0)), 0.0
      ]
  }, {
      "lambdaweight_fn":
          lambdaweights.dcg2_lambdaweight,
      "expected": [
          0.0,
          3.0 * abs(discount(1) - discount(2)) * abs(gain(0.0) - gain(1.0)),
          3.0 * abs(discount(2) - discount(3)) * abs(gain(0.0) - gain(0.3)),
          3.0 * abs(discount(1) - discount(2)) * abs(gain(1.0) - gain(0.0)),
          0.0,
          3.0 * abs(discount(1) - discount(2)) * abs(gain(1.0) - gain(0.3)),
          3.0 * abs(discount(2) - discount(3)) * abs(gain(0.3) - gain(0.0)),
          3.0 * abs(discount(1) - discount(2)) * abs(gain(0.3) - gain(1.0)), 0.0
      ]
  }])
  def test_computes_lambdaweights(self, lambdaweight_fn, expected):
    scores = jnp.array([0.0, 1.0, 2.0])
    labels = jnp.array([0.0, 1.0, 0.3])

    result = lambdaweight_fn(scores, labels)

    np.testing.assert_allclose(result, expected, rtol=1e-5)

  @parameterized.parameters([{
      "lambdaweight_fn":
          lambdaweights.dcg_lambdaweight,
      "normalizer": [
          [
              gain(1.0) * discount(1) + gain(0.3) * discount(2) +
              gain(0.0) * discount(3)
          ], [
              gain(2.0) * discount(1) + gain(1.0) * discount(2) +
              gain(0.0) * discount(3)
          ]
      ]
  }, {
      "lambdaweight_fn":
          lambdaweights.dcg2_lambdaweight,
      "normalizer": [
          [
              gain(1.0) * discount(1) + gain(0.3) * discount(2) +
              gain(0.0) * discount(3)
          ], [
              gain(2.0) * discount(1) + gain(1.0) * discount(2) +
              gain(0.0) * discount(3)
          ]
      ]
  }])  # pyformat: disable
  def test_computes_normalized_lambdaweights(self, lambdaweight_fn, normalizer):
    scores = jnp.array([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]])
    labels = jnp.array([[0.0, 1.0, 0.3], [1.0, 0.0, 2.0]])

    result = lambdaweight_fn(scores, labels, normalize=True)
    result_unnormalized = lambdaweight_fn(scores, labels)

    np.testing.assert_allclose(
        result, result_unnormalized / jnp.array(normalizer), rtol=1e-5)

  @parameterized.parameters([{
      "lambdaweight_fn":
          lambdaweights.labeldiff_lambdaweight,
      "expected": [[0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]
  }, {
      "lambdaweight_fn":
          lambdaweights.dcg_lambdaweight,
      "expected": [
          [
              0.0,
              3.0 * abs(discount(2) - discount(3)) * abs(gain(0.0) - gain(2.0)),
              3.0 * abs(discount(1) - discount(3)) * abs(gain(0.0) - gain(1.0)),
              3.0 * abs(discount(3) - discount(2)) * abs(gain(2.0) - gain(0.0)),
              0.0,
              3.0 * abs(discount(1) - discount(2)) * abs(gain(2.0) - gain(1.0)),
              3.0 * abs(discount(3) - discount(1)) * abs(gain(1.0) - gain(0.0)),
              3.0 * abs(discount(2) - discount(1)) * abs(gain(1.0) - gain(2.0)),
              0.0
          ],
          [
              0.0, 0.0,
              3.0 * abs(discount(2) - discount(3)) * abs(gain(1.0) - gain(0.0)),
              0.0, 0.0,
              3.0 * abs(discount(2) - discount(1)) * abs(gain(0.0) - gain(1.0)),
              3.0 * abs(discount(3) - discount(2)) * abs(gain(1.0) - gain(0.0)),
              3.0 * abs(discount(1) - discount(2)) * abs(gain(1.0) - gain(0.0)),
              0.0
          ],
      ]
  }, {
      "lambdaweight_fn":
          lambdaweights.dcg2_lambdaweight,
      "expected": [
          [
              0.0,
              3.0 * abs(discount(1) - discount(2)) * abs(gain(0.0) - gain(2.0)),
              3.0 * abs(discount(2) - discount(3)) * abs(gain(0.0) - gain(1.0)),
              3.0 * abs(discount(1) - discount(2)) * abs(gain(2.0) - gain(0.0)),
              0.0,
              3.0 * abs(discount(1) - discount(2)) * abs(gain(2.0) - gain(1.0)),
              3.0 * abs(discount(2) - discount(3)) * abs(gain(1.0) - gain(0.0)),
              3.0 * abs(discount(1) - discount(2)) * abs(gain(1.0) - gain(2.0)),
              0.0
          ],
          [
              0.0, 0.0,
              3.0 * abs(discount(1) - discount(2)) * abs(gain(1.0) - gain(0.0)),
              0.0, 0.0,
              3.0 * abs(discount(1) - discount(2)) * abs(gain(0.0) - gain(1.0)),
              3.0 * abs(discount(1) - discount(2)) * abs(gain(1.0) - gain(0.0)),
              3.0 * abs(discount(1) - discount(2)) * abs(gain(1.0) - gain(0.0)),
              0.0
          ],
      ]
  }])
  def test_lambdaweights_with_batchdim_or_segments(
      self, lambdaweight_fn, expected
  ):
    scores = jnp.array([[0.0, 1.0, 2.0], [0.5, 1.5, 1.0]])
    labels = jnp.array([[0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])

    result = lambdaweight_fn(scores, labels)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    segmented_scores = jnp.array([0.0, 1.0, 2.0])
    segmented_labels = jnp.array([0.0, 2.0, 1.0])
    segments = jnp.array([1, 1, 1])

    segmented_result = lambdaweight_fn(
        segmented_scores, segmented_labels, segments=segments
    )
    np.testing.assert_allclose(segmented_result, expected[0], rtol=1e-5)

  @parameterized.parameters([
      lambdaweights.labeldiff_lambdaweight,
      lambdaweights.dcg_lambdaweight,
      lambdaweights.dcg2_lambdaweight,
  ])
  def test_lambdaweights_with_empty_list(self, lambdaweight_fn):
    scores = jnp.array([])
    labels = jnp.array([])

    expected = jnp.array([])
    result = lambdaweight_fn(scores, labels)

    np.testing.assert_allclose(result, expected)

  @parameterized.parameters([{
      "lambdaweight_fn": lambdaweights.labeldiff_lambdaweight,
      "expected": [0.0, 1.0, 0.3, 1.0, 0.0, 0.7, 0.3, 0.7, 0.0]
  }, {
      "lambdaweight_fn":
          lambdaweights.dcg_lambdaweight,
      "expected": [
          0.0, 0.0,
          3.0 * abs(discount(1) - discount(2)) * abs(gain(0.0) - gain(0.3)),
          0.0, 0.0, 0.0,
          3.0 * abs(discount(2) - discount(1)) * abs(gain(0.3) - gain(0.0)),
          0.0, 0.0
      ]
  }, {
      "lambdaweight_fn":
          lambdaweights.dcg2_lambdaweight,
      "expected": [
          0.0, 0.0,
          3.0 * abs(discount(1) - discount(2)) * abs(gain(0.0) - gain(0.3)),
          0.0, 0.0, 0.0,
          3.0 * abs(discount(1) - discount(2)) * abs(gain(0.3) - gain(0.0)),
          0.0, 0.0
      ]
  }])
  def test_lambdaweights_with_where_mask(self, lambdaweight_fn, expected):
    scores = jnp.array([0.0, 1.0, 2.0])
    labels = jnp.array([0.0, 1.0, 0.3])
    where = jnp.array([1, 0, 1], dtype=jnp.bool_)

    result = lambdaweight_fn(scores, labels, where=where)

    np.testing.assert_allclose(result, expected, rtol=1e-5)

  @parameterized.parameters([{
      "lambdaweight_fn": lambdaweights.labeldiff_lambdaweight,
      "expected": [0.0, 1.0, 0.3, 1.0, 0.0, 0.7, 0.3, 0.7, 0.0]
  }, {
      "lambdaweight_fn":
          lambdaweights.dcg_lambdaweight,
      "expected": [
          0.0,
          3.0 * abs(discount(2) - discount(3)) * abs(0.0 - 0.5 * gain(1.0)),
          3.0 * abs(discount(1) - discount(3)) *
          abs(1.5 * gain(0.0) - gain(0.3)),
          3.0 * abs(discount(3) - discount(2)) * abs(0.5 * gain(1.0) - 0.0),
          0.0, 3.0 * abs(discount(1) - discount(2)) *
          abs(0.5 * gain(1.0) - gain(0.3)), 3.0 *
          abs(discount(3) - discount(1)) * abs(gain(0.3) - 1.5 * gain(0.0)),
          3.0 * abs(discount(2) - discount(1)) *
          abs(gain(0.3) - 0.5 * gain(1.0)), 0.0
      ]
  }, {
      "lambdaweight_fn":
          lambdaweights.dcg2_lambdaweight,
      "expected": [
          0.0, 3.0 * abs(discount(1) - discount(2)) *
          abs(gain(0.0) - 0.5 * gain(1.0)), 3.0 *
          abs(discount(2) - discount(3)) * abs(1.5 * gain(0.0) - gain(0.3)),
          3.0 * abs(discount(1) - discount(2)) *
          abs(0.5 * gain(1.0) - gain(0.0)), 0.0, 3.0 *
          abs(discount(1) - discount(2)) * abs(0.5 * gain(1.0) - gain(0.3)),
          3.0 * abs(discount(2) - discount(3)) *
          abs(gain(0.3) - 1.5 * gain(0.0)), 3.0 *
          abs(discount(2) - discount(1)) * abs(gain(0.3) - 0.5 * gain(1.0)), 0.0
      ]
  }])
  def test_lambdaweights_with_weights(self, lambdaweight_fn, expected):
    scores = jnp.array([0.0, 1.0, 2.0])
    labels = jnp.array([0.0, 1.0, 0.3])
    weights = jnp.array([1.5, 0.5, 1.0])

    result = lambdaweight_fn(scores, labels, weights=weights)

    np.testing.assert_allclose(result, expected, rtol=1e-5)

  @parameterized.parameters([{
      "lambdaweight_fn":
          lambdaweights.dcg_lambdaweight,
      "expected": [
          0.0, 0.0, 3.0 * abs(discount(1) - 0.0) * abs(gain(0.0) - gain(0.3)),
          0.0, 0.0, 3.0 * abs(discount(1) - 0.0) * abs(gain(1.0) - gain(0.3)),
          3.0 * abs(0.0 - discount(1)) * abs(gain(0.3) - gain(0.0)),
          3.0 * abs(0.0 - discount(1)) * abs(gain(0.3) - gain(1.0)), 0.0
      ]
  }, {
      "lambdaweight_fn":
          lambdaweights.dcg2_lambdaweight,
      "expected": [
          0.0, 3.0 * (1.0 / (1.0 - discount(3))) *
          abs(discount(1) - discount(2)) * abs(gain(0.0) - gain(1.0)),
          3.0 * (1.0 / (1.0 - discount(3))) * abs(discount(2) - discount(3)) *
          abs(gain(0.0) - gain(0.3)), 3.0 * (1.0 / (1.0 - discount(3))) *
          abs(discount(1) - discount(2)) * abs(gain(1.0) - gain(0.0)), 0.0,
          3.0 * (1.0 / (1.0 - discount(2))) * abs(discount(1) - discount(2)) *
          abs(gain(1.0) - gain(0.3)), 3.0 * (1.0 / (1.0 - discount(3))) *
          abs(discount(2) - discount(3)) * abs(gain(0.3) - gain(0.0)),
          3.0 * (1.0 / (1.0 - discount(2))) * abs(discount(2) - discount(1)) *
          abs(gain(0.3) - gain(1.0)), 0.0
      ]
  }])
  def test_lambdaweights_with_topn(self, lambdaweight_fn, expected):
    scores = jnp.array([0.0, 1.0, 2.0])
    labels = jnp.array([0.0, 1.0, 0.3])
    topn = 1

    result = lambdaweight_fn(scores, labels, topn=topn)

    np.testing.assert_allclose(result, expected, rtol=1e-5)

  @parameterized.parameters([{
      "loss_fn": losses.pairwise_hinge_loss,
      "lambdaweight_fn": lambdaweights.labeldiff_lambdaweight,
      "expected": 0.63333327
  }, {
      "loss_fn": losses.pairwise_logistic_loss,
      "lambdaweight_fn": lambdaweights.labeldiff_lambdaweight,
      "expected": 0.6564648
  }, {
      "loss_fn": losses.pairwise_hinge_loss,
      "lambdaweight_fn": lambdaweights.dcg_lambdaweight,
      "expected": 0.43137252 * 4.0
  }, {
      "loss_fn": losses.pairwise_logistic_loss,
      "lambdaweight_fn": lambdaweights.dcg_lambdaweight,
      "expected": 0.34675273 * 4.0
  }, {
      "loss_fn": losses.pairwise_hinge_loss,
      "lambdaweight_fn": lambdaweights.dcg2_lambdaweight,
      "expected": 0.45518658 * 4.0
  }, {
      "loss_fn": losses.pairwise_logistic_loss,
      "lambdaweight_fn": lambdaweights.dcg2_lambdaweight,
      "expected": 0.4053712 * 4.0
  }, {
      "loss_fn": losses.pairwise_mse_loss,
      "lambdaweight_fn": lambdaweights.dcg_lambdaweight,
      "expected": 0.61966689551 * 4.0
  }])
  def test_computes_with_pairwise_loss(self, loss_fn, lambdaweight_fn,
                                       expected):
    scores = jnp.array([0.3, 1.9, 1.5, 1.2])
    labels = jnp.array([0.0, 1.0, 1.0, 2.0])
    where = jnp.array([1, 1, 0, 1], dtype=jnp.bool_)

    result = loss_fn(
        scores, labels, where=where, lambdaweight_fn=lambdaweight_fn)

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def load_tests(loader, tests, ignore):
  del loader, ignore  # Unused.
  tests.addTests(
      doctest.DocTestSuite(
          lambdaweights, globs={
              "jax": jax,
              "jnp": jnp,
              "rax": rax
          }))
  return tests


if __name__ == "__main__":
  absltest.main()
