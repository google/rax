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
"""Tests for rax._src.weights."""

import doctest

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

import rax
from rax._src import lambdaweights
from rax._src import losses


class LambdaweightsTest(parameterized.TestCase):

  @parameterized.parameters([{
      "lambdaweight_fn": lambdaweights.labeldiff_lambdaweight,
      "expected": [0.0, 1.0, 0.3, 1.0, 0.0, 0.7, 0.3, 0.7, 0.0]
  }])
  def test_computes_lambdaweights(self, lambdaweight_fn, expected):
    scores = jnp.array([0.0, 1.0, 2.0])
    labels = jnp.array([0.0, 1.0, 0.3])

    result = lambdaweight_fn(scores, labels)

    np.testing.assert_allclose(result, expected)

  @parameterized.parameters([{
      "lambdaweight_fn":
          lambdaweights.labeldiff_lambdaweight,
      "expected": [[0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]
  }])
  def test_lambdaweights_with_batchdim(self, lambdaweight_fn, expected):
    scores = jnp.array([[0.0, 1.0, 2.0], [0.5, 1.5, 1.0]])
    labels = jnp.array([[0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])

    result = lambdaweight_fn(scores, labels)

    np.testing.assert_allclose(result, expected)

  @parameterized.parameters([lambdaweights.labeldiff_lambdaweight])
  def test_lambdaweights_with_empty_list(self, lambdaweight_fn):
    scores = jnp.array([])
    labels = jnp.array([])

    expected = jnp.array([])
    result = lambdaweight_fn(scores, labels)

    np.testing.assert_allclose(result, expected)

  @parameterized.parameters([{
      "lambdaweight_fn": lambdaweights.labeldiff_lambdaweight,
      "expected": [0.0, 1.0, 0.3, 1.0, 0.0, 0.7, 0.3, 0.7, 0.0]
  }])
  def test_lambdaweights_with_where_mask(self, lambdaweight_fn, expected):
    scores = jnp.array([0.0, 1.0, 2.0])
    labels = jnp.array([0.0, 1.0, 0.3])
    where = jnp.array([1, 0, 1], dtype=jnp.bool_)

    result = lambdaweight_fn(scores, labels, where=where)

    np.testing.assert_allclose(result, expected)

  @parameterized.parameters([{
      "lambdaweight_fn": lambdaweights.labeldiff_lambdaweight,
      "expected": [0.0, 1.0, 0.3, 1.0, 0.0, 0.7, 0.3, 0.7, 0.0]
  }])
  def test_lambdaweights_with_weights(self, lambdaweight_fn, expected):
    scores = jnp.array([0.0, 1.0, 2.0])
    labels = jnp.array([0.0, 1.0, 0.3])
    weights = jnp.array([1.5, 0.5, 1.0])

    result = lambdaweight_fn(scores, labels, weights=weights)

    np.testing.assert_allclose(result, expected)

  @parameterized.parameters([{
      "loss_fn": losses.pairwise_hinge_loss,
      "lambdaweight_fn": lambdaweights.labeldiff_lambdaweight,
      "expected": 0.63333327
  }, {
      "loss_fn": losses.pairwise_logistic_loss,
      "lambdaweight_fn": lambdaweights.labeldiff_lambdaweight,
      "expected": 0.6564648
  }])
  def test_computes_with_pairwise_loss(self, loss_fn, lambdaweight_fn,
                                       expected):
    scores = jnp.array([0.3, 1.9, 1.2])
    labels = jnp.array([0.0, 1.0, 2.0])

    result = loss_fn(scores, labels, lambdaweight_fn=lambdaweight_fn)

    np.testing.assert_allclose(result, expected)


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
