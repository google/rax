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
"""Tests for rax._src.losses."""

import doctest
import functools
import math
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

import rax
from rax._src import losses

# Export symbols from math for conciser test value definitions.
exp = math.exp
log = math.log
logloss = lambda x: log(1. + exp(-x))
sigmoid = lambda x: 1. / (1. + exp(-x))


class LossesTest(parameterized.TestCase):

  @parameterized.parameters([{
      "loss_fn":
          losses.softmax_loss,
      "expected_value":
          -(log(exp(2.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.))) +
            log(exp(1.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.))))
  }, {
      "loss_fn":
          losses.listmle_loss,
      "expected_value":
          -sum([
              log(exp(1.) / (exp(1.) + exp(2.) + exp(0.) + exp(3.))),
              log(exp(2.) / (exp(2.) + exp(0.) + exp(3.))),
              log(exp(0.) / (exp(0.) + exp(3.))),
              log(exp(3.) / (exp(3.))),
          ])
  }, {
      "loss_fn":
          losses.poly1_softmax_loss,
      "expected_value":
          -(log(exp(2.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.))) +
            log(exp(1.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.)))) +
          (1. - (0.5 * exp(2.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.)) + 0.5 *
                 (exp(1.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.)))))
  }, {
      "loss_fn":
          losses.unique_softmax_loss,
      "expected_value":
          -(log(exp(2.) / (exp(0.) + exp(3.) + exp(2.))) +
            log(exp(1.) / (exp(0.) + exp(3.) + exp(1.))))
  }, {
      "loss_fn":
          functools.partial(losses.poly1_softmax_loss, epsilon=0.1),
      "expected_value":
          -(log(exp(2.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.))) +
            log(exp(1.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.)))) + 0.1 *
          (1. - (0.5 * exp(2.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.)) + 0.5 *
                 (exp(1.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.)))))
  }, {
      "loss_fn": losses.pairwise_hinge_loss,
      "expected_value": (3. - 1. + 1.) + (3. - 2. + 1.)
  }, {
      "loss_fn":
          losses.pairwise_logistic_loss,
      "expected_value":
          logloss(1. - 0.) + logloss(1. - 3.) + logloss(2. - 3.) +
          logloss(2. - 0.)
  }, {
      "loss_fn":
          losses.pairwise_soft_zero_one_loss,
      "expected_value":
          sigmoid(-(1. - 0.)) + sigmoid(-(1. - 3.)) + sigmoid(-(2. - 3.)) +
          sigmoid(-(2. - 0.))
  }, {
      "loss_fn":
          losses.pointwise_sigmoid_loss,
      "expected_value":
          -log(1. - sigmoid(0.)) - log(1. - sigmoid(3.)) - log(sigmoid(1.)) -
          log(sigmoid(2.))
  }, {
      "loss_fn":
          losses.pointwise_mse_loss,
      "expected_value":
          (0. - 0.)**2 + (3. - 0.)**2 + (1. - 1.)**2 + (2. - 1.)**2
  }, {
      "loss_fn":
          losses.pairwise_mse_loss,
      "expected_value":
          ((0. - 3.) - (0. - 0.))**2 + ((0. - 1.) - (0. - 1.))**2 +
          ((0. - 2.) - (0. - 1.))**2 + ((3. - 0.) - (0. - 0.))**2 +
          ((3. - 1.) - (0. - 1.))**2 + ((3. - 2.) - (0. - 1.))**2 +
          ((1. - 0.) - (1. - 0.))**2 + ((1. - 3.) - (1. - 0.))**2 +
          ((1. - 2.) - (1. - 1.))**2 + ((2. - 0.) - (1. - 0.))**2 +
          ((2. - 3.) - (1. - 0.))**2 + ((2. - 1.) - (1. - 1.))**2
  }, {
      "loss_fn":
          losses.pairwise_qr_loss,
      "expected_value":
          0.5 * (((1. - 0.) - (1. - 0.)) + ((1. - 0.) - (1. - 3.)) +
                 ((1. - 0.) - (2. - 3.))) + 0.5 * (((2. - 0.) - (1. - 0.)))
  }, {
      "loss_fn":
          functools.partial(losses.pairwise_qr_loss, tau=1.0),
      "expected_value":
          1. * (((1. - 0.) - (1. - 0.)) + ((1. - 0.) - (1. - 3.)) +
                ((1. - 0.) - (2. - 3.))) + 0. * (((2. - 0.) - (1. - 0.)))
  }, {
      "loss_fn":
          functools.partial(losses.pairwise_qr_loss, squared=True),
      "expected_value":
          0.5 * (((1. - 0.) - (1. - 0.))**2 + ((1. - 0.) - (1. - 3.))**2 +
                 ((1. - 0.) - (2. - 3.))**2) + 0.5 * (((2. - 0.) -
                                                       (1. - 0.))**2)
  }])
  def test_computes_loss_value(self, loss_fn, expected_value):
    scores = jnp.asarray([0., 3., 1., 2.])
    labels = jnp.asarray([0., 0., 1., 1.])

    loss = loss_fn(scores, labels, reduce_fn=jnp.sum)

    np.testing.assert_allclose(jnp.asarray(expected_value), loss)

  @parameterized.parameters([{
      "loss_fn":
          losses.softmax_loss,
      "expected_value":
          -((-2.1e26 - (0. + -2.1e26 + 3.4e37 + 42.)) +
            (3.4e37 - (0. + -2.1e26 + 3.4e37 + 42.)))
  }, {
      "loss_fn": losses.listmle_loss,
      "expected_value": 3.4e37
  }, {
      "loss_fn":
          losses.poly1_softmax_loss,
      "expected_value":
          -((-2.1e26 - (0. + -2.1e26 + 3.4e37 + 42.)) +
            (3.4e37 - (0. + -2.1e26 + 3.4e37 + 42.)))
  }, {
      "loss_fn":
          losses.unique_softmax_loss,
      "expected_value":
          -((-2.1e26 - (0. + 42.)) + (3.4e37 - (0. + 3.4e37 + 42.)))
  }, {
      "loss_fn": losses.pairwise_hinge_loss,
      "expected_value": (1. - (-2.1e26 - 0.)) + (1. - (-2.1e26 - 42.0))
  }, {
      "loss_fn": losses.pairwise_logistic_loss,
      "expected_value": 2.1e26 + 2.1e26
  }, {
      "loss_fn": losses.pairwise_soft_zero_one_loss,
      "expected_value": 1.0 + 1.0
  }, {
      "loss_fn": losses.pointwise_sigmoid_loss,
      "expected_value": 2.1e26 - log(1. - sigmoid(0.)) + 42.0
  }, {
      "loss_fn":
          losses.pointwise_mse_loss,
      "expected_value":
          (0. - 0.)**2 + (-2.1e26 - 1.)**2 + (3.4e37 - 1.)**2 + (42. - 0.)**2
  }, {
      "loss_fn":
          losses.pairwise_mse_loss,
      "expected_value":
          (2.1e26 - -1.)**2 + (-3.4e37 - -1.)**2 + (-42. - 0.)**2 +
          (-2.1e26 - 1.)**2 + ((-2.1e26 - 3.4e37) - 0.)**2 +
          ((-2.1e26 - 42.) - 1.)**2 + (3.4e37 - 1.)**2 +
          ((3.4e37 - -2.1e26) - 0.)**2 + ((3.4e37 - 42.) - 1.)**2 +
          (42. - 0.)**2 + ((42. - -2.1e26) - -1.)**2 + ((42. - 3.4e37) - -1.)**2
  }, {
      "loss_fn":
          losses.pairwise_qr_loss,
      "expected_value":
          0.5 * (((1. - 0.) - (-2.1e26 - 0.)) + ((1. - 0.) - (-2.1e26 - 42.))) +
          0.5 * (((3.4e37 - 0.) - (1. - 0.)) + ((3.4e37 - 42.) - (1. - 0.)))
  }])
  def test_computes_loss_with_extreme_inputs(self, loss_fn, expected_value):
    scores = jnp.asarray([0., -2.1e26, 3.4e37, 42.0])
    labels = jnp.asarray([0., 1., 1., 0.])

    loss = loss_fn(scores, labels, reduce_fn=jnp.sum)

    np.testing.assert_allclose(jnp.asarray(expected_value), loss)

  @parameterized.parameters([{
      "loss_fn": losses.softmax_loss,
      "expected_value": 0.
  }, {
      "loss_fn":
          losses.listmle_loss,
      "expected_value":
          -sum([
              log(exp(0.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.))),
              log(exp(3.) / (exp(3.) + exp(1.) + exp(2.))),
              log(exp(1.) / (exp(1.) + exp(2.))),
              log(exp(2.) / (exp(2.))),
          ])
  }, {
      "loss_fn":
          losses.poly1_softmax_loss,
      "expected_value":
          1. - sum([
              0.25 * (exp(0.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.))),
              0.25 * (exp(3.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.))),
              0.25 * (exp(1.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.))),
              0.25 * (exp(2.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.))),
          ])
  }, {
      "loss_fn": losses.unique_softmax_loss,
      "expected_value": 0.
  }, {
      "loss_fn": losses.pairwise_hinge_loss,
      "expected_value": 0.
  }, {
      "loss_fn": losses.pairwise_logistic_loss,
      "expected_value": 0.
  }, {
      "loss_fn": losses.pairwise_soft_zero_one_loss,
      "expected_value": 0.
  }, {
      "loss_fn":
          losses.pointwise_sigmoid_loss,
      "expected_value":
          -log(1. - sigmoid(0.)) - log(1. - sigmoid(3.)) -
          log(1. - sigmoid(1.)) - log(1. - sigmoid(2.))
  }, {
      "loss_fn":
          losses.pointwise_mse_loss,
      "expected_value":
          (0. - 0.)**2 + (3. - 0.)**2 + (1. - 0.)**2 + (2. - 0.)**2
  }, {
      "loss_fn":
          losses.pairwise_mse_loss,
      "expected_value": (-3.)**2 + (-1.)**2 + (-2.)**2 + 3.**2 + 2.**2 + 1.**2 +
                        1.**2 + (-2.)**2 + (-1.)**2 + 2.**2 + (-1.)**2 + 1.**2
  }, {
      "loss_fn": losses.pairwise_qr_loss,
      "expected_value": 0.
  }])
  def test_computes_loss_for_zero_labels(self, loss_fn, expected_value):
    scores = jnp.asarray([0., 3., 1., 2.])
    labels = jnp.asarray([0., 0., 0., 0.])

    loss = loss_fn(scores, labels, reduce_fn=jnp.sum)

    np.testing.assert_allclose(jnp.asarray(expected_value), loss)

  @parameterized.parameters([{
      "loss_fn":
          losses.softmax_loss,
      "expected_value":
          -(2. * log(exp(2.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.))) +
            log(exp(1.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.))))
  }, {
      "loss_fn":
          losses.poly1_softmax_loss,
      "expected_value":
          -(2. * log(exp(2.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.))) +
            log(exp(1.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.)))) +
          (1. - (2. / 3. * exp(2.) /
                 (exp(0.) + exp(3.) + exp(1.) + exp(2.)) + 1. / 3. *
                 (exp(1.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.)))))
  }, {
      "loss_fn":
          losses.unique_softmax_loss,
      "expected_value":
          -(2. * log(exp(2.) / (exp(0.) + exp(3.) + exp(2.))) +
            log(exp(1.) / (exp(0.) + exp(3.) + exp(1.))))
  }, {
      "loss_fn": losses.pairwise_hinge_loss,
      "expected_value": 7.
  }, {
      "loss_fn": losses.pairwise_logistic_loss,
      "expected_value": 5.320569
  }, {
      "loss_fn": losses.pairwise_soft_zero_one_loss,
      "expected_value": 2.850261
  }, {
      "loss_fn":
          losses.pointwise_sigmoid_loss,
      "expected_value":
          -log(1. - sigmoid(0.)) - log(1. - sigmoid(3.)) -
          2. * log(sigmoid(2.)) - log(sigmoid(1.))
  }, {
      "loss_fn":
          losses.pointwise_mse_loss,
      "expected_value":
          (0. - 0.)**2 + (3. - 0.)**2 + 2. * (2. - 1.)**2 + (1. - 1.)**2
  }, {
      "loss_fn":
          losses.pairwise_mse_loss,
      "expected_value":
          (1. * ((-3. - 0.)**2 + (-2. - -1.)**2 + (-1. - -1.)**2)) +
          (1. * ((3. - 0.)**2 + (1. - -1.)**2 + (2. - -1.)**2)) +
          (2. * ((2. - 1.)**2 + (-1. - 1.)**2 + (1. - 0.)**2)) +
          (1. * ((1. - 1.)**2 + (-2. - 1.)**2 + (-1. - 0.)**2))
  }])
  def test_computes_weighted_loss_value(self, loss_fn, expected_value):
    scores = jnp.asarray([0., 3., 2., 1.])
    labels = jnp.asarray([0., 0., 1., 1.])
    weights = jnp.asarray([1., 1., 2., 1.])

    loss = loss_fn(scores, labels, weights=weights, reduce_fn=jnp.sum)

    np.testing.assert_allclose(jnp.asarray(expected_value), loss, rtol=1e-6)

  @parameterized.parameters([{
      "loss_fn":
          losses.softmax_loss,
      "expected_value": [
          -(log(exp(2.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.))) +
            log(exp(1.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.)))),
          -(2. * log(exp(3.) / (exp(3.) + exp(1.) + exp(4.) + exp(2.))) +
            log(exp(4.) / (exp(3.) + exp(1.) + exp(4.) + exp(2.))))
      ]
  }, {
      "loss_fn":
          losses.listmle_loss,
      "expected_value": [
          -sum([
              log(exp(1.) / (exp(1.) + exp(2.) + exp(0.) + exp(3.))),
              log(exp(2.) / (exp(2.) + exp(0.) + exp(3.))),
              log(exp(0.) / (exp(0.) + exp(3.))),
              log(exp(3.) / (exp(3.))),
          ]), -sum([
              log(exp(3.) / (exp(3.) + exp(4.) + exp(1.) + exp(2.))),
              log(exp(4.) / (exp(4.) + exp(1.) + exp(2.))),
              log(exp(1.) / (exp(1.) + exp(2.))),
              log(exp(2.) / (exp(2.))),
          ])
      ]
  }, {
      "loss_fn":
          losses.poly1_softmax_loss,
      "expected_value": [
          -(log(exp(2.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.))) +
            log(exp(1.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.)))) +
          (1. - (0.5 * (exp(2.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.))) +
                 0.5 * (exp(1.) / (exp(0.) + exp(3.) + exp(1.) + exp(2.))))),
          -(2. * log(exp(3.) / (exp(3.) + exp(1.) + exp(4.) + exp(2.))) +
            log(exp(4.) / (exp(3.) + exp(1.) + exp(4.) + exp(2.)))) +
          (1. - (2. / 3. * (exp(3.) / (exp(3.) + exp(1.) + exp(4.) + exp(2.))) +
                 1. / 3. * (exp(4.) / (exp(3.) + exp(1.) + exp(4.) + exp(2.)))))
      ]
  }, {
      "loss_fn":
          losses.unique_softmax_loss,
      "expected_value": [
          -(log(exp(2.) / (exp(0.) + exp(3.) + exp(2.))) +
            log(exp(1.) / (exp(0.) + exp(3.) + exp(1.)))),
          -(3. * log(exp(3.) / (exp(3.) + exp(1.) + exp(4.) + exp(2.))) +
            log(exp(4.) / (exp(1.) + exp(4.) + exp(2.))))
      ]
  }, {
      "loss_fn": losses.pairwise_hinge_loss,
      "expected_value": [(3. - 1. + 1.) + (3. - 2. + 1.), (4. - 3. + 1.)]
  }, {
      "loss_fn":
          losses.pairwise_logistic_loss,
      "expected_value": [
          logloss(1. - 0.) + logloss(1. - 3.) + logloss(2. - 3.) +
          logloss(2. - 0.),
          logloss(3. - 1.) + logloss(3. - 4.) + logloss(3. - 2.) +
          logloss(4. - 1.) + logloss(4. - 2.)
      ]
  }, {
      "loss_fn":
          losses.pointwise_sigmoid_loss,
      "expected_value": [
          -log(1. - sigmoid(0.)) - log(1. - sigmoid(3.)) - log(sigmoid(1.)) -
          log(sigmoid(2.)),
          -log(sigmoid(3.)) - log(1. - sigmoid(1.)) - log(sigmoid(4.)) -
          log(1. - sigmoid(2.))
      ]
  }, {
      "loss_fn":
          losses.pointwise_mse_loss,
      "expected_value": [
          (0. - 0.)**2 + (3. - 0.)**2 + (1. - 1.)**2 + (2. - 1.)**2,
          (3. - 2.)**2 + (1. - 0.)**2 + (4. - 1.)**2 + (2. - 0.)**2
      ]
  }, {
      "loss_fn":
          losses.pairwise_mse_loss,
      "expected_value": [
          (-3. - 0.)**2 + (-1. - -1.)**2 + (-2. - -1.)**2 +
          (3. - 0.)**2 + (2. - -1.)**2 + (1. - -1.)**2 +
          (1. - 1.)**2 + (-2. - 1.)**2 + (-1. - 0.)**2 +
          (2. - 1.)**2 + (-1. - 1.)**2 + (1. - 0.)**2,
          (2. - 2.)**2 + (-1. - 1.)**2 + (1. - 2.)**2 +
          (-2. - -2.)**2 + (-3. - -1.)**2 + (-1. - 0.)**2 +
          (1. - -1.)**2 + (3. - 1.)**2 + (2. - 1.)**2 +
          (-1. - -2.)**2 + (1. - 0.)**2 + (-2. - -1.)**2
      ]
  }])  # pyformat: disable
  def test_computes_loss_value_with_vmap(self, loss_fn, expected_value):
    scores = jnp.asarray([[0., 3., 1., 2.], [3., 1., 4., 2.]])
    labels = jnp.asarray([[0., 0., 1., 1.], [2., 0., 1., 0.]])

    loss_fn = functools.partial(loss_fn, reduce_fn=jnp.sum)
    vmap_loss_fn = jax.vmap(loss_fn, in_axes=(0, 0), out_axes=0)
    loss = vmap_loss_fn(scores, labels)

    np.testing.assert_allclose(jnp.asarray(expected_value), loss)

  @parameterized.parameters([{
      "loss_fn": losses.softmax_loss,
      "expected_value": [
          -log(exp(2.) / (exp(2.) + exp(1.) + exp(3.))),
          -log(exp(1.5) / (exp(1.) + exp(0.5) + exp(1.5)))
      ],
      "normalizer": 2.
  }, {
      "loss_fn": losses.listmle_loss,
      "expected_value": [
          -sum([
              log(exp(2.) / (exp(2.) + exp(1.) + exp(3.))),
              log(exp(1.) / (exp(1.) + exp(3.))),
              log(exp(3.) / (exp(3.))),
          ]), -sum([
              log(exp(1.5) / (exp(1.5) + exp(1.) + exp(0.5))),
              log(exp(1.) / (exp(1.) + exp(0.5))),
              log(exp(0.5) / (exp(0.5))),
          ])
      ],
      "normalizer": 2.
  }, {
      "loss_fn": losses.poly1_softmax_loss,
      "expected_value": [
          -log(exp(2.) / (exp(2.) + exp(1.) + exp(3.))) +
          (1. - (exp(2.) / (exp(2.) + exp(1.) + exp(3.)))),
          -log(exp(1.5) / (exp(1.) + exp(0.5) + exp(1.5))) +
          (1. - (exp(1.5) / (exp(1.) + exp(0.5) + exp(1.5))))
      ],
      "normalizer": 2.
  }, {
      "loss_fn": losses.unique_softmax_loss,
      "expected_value": [
          -log(exp(2.) / (exp(2.) + exp(1.) + exp(3.))),
          -log(exp(1.5) / (exp(1.) + exp(0.5) + exp(1.5)))
      ],
      "normalizer": 2.
  }, {
      "loss_fn": losses.pairwise_hinge_loss,
      "expected_value": [2., .5],
      "normalizer": 4.
  }, {
      "loss_fn": losses.pairwise_logistic_loss,
      "expected_value": [
          logloss(2. - 1.) + logloss(2. - 3.),
          logloss(1.5 - 1.) + logloss(1.5 - 0.5)
      ],
      "normalizer": 4.
  }, {
      "loss_fn": losses.pairwise_soft_zero_one_loss,
      "expected_value": [
          sigmoid(-(2. - 1.)) + sigmoid(-(2. - 3.)),
          sigmoid(-(1.5 - 1.)) + sigmoid(-(1.5 - 0.5))
      ],
      "normalizer": 4.
  }, {
      "loss_fn": losses.pointwise_sigmoid_loss,
      "expected_value": [
          -log(sigmoid(2.)) - log(1. - sigmoid(1.)) - log(1. - sigmoid(3.)),
          -log(sigmoid(1.5)) - log(1. - sigmoid(1.)) - log(1. - sigmoid(0.5))
      ],
      "normalizer": 6.
  }, {
      "loss_fn": losses.pointwise_mse_loss,
      "expected_value": [(2. - 1.)**2 + (1. - 0.)**2 + (3. - 0.)**2,
                         (1. - 0.)**2 + (0.5 - 0.)**2 + (1.5 - 1.)**2],
      "normalizer": 6.
  }, {
      "loss_fn": losses.pairwise_mse_loss,
      "expected_value": [(1. - 1.)**2 + (-1. - 1.)**2 + (-1. - -1.)**2 +
                         (-2. - 0.)**2 + (1. - -1.)**2 + (2. - 0.)**2,
                         (0.5 - 0.)**2 + (-0.5 - -1.)**2 + (-0.5 - 0.)**2 +
                         (-1. - -1.)**2 + (0.5 - 1.)**2 + (1. - 1.)**2],
      "normalizer": 9. + 9.
  }, {
      "loss_fn": losses.pairwise_qr_loss,
      "expected_value": [
          0.5 * (((1. - 0.) - (2. - 1.)) + ((1. - 0.) - (2. - 3.))),
          0.5 * (((1. - 0.) - (1.5 - 1.)) + ((1. - 0.) - (1.5 - 0.5)))
      ],
      "normalizer": 4.
  }])
  def test_computes_reduced_loss(self, loss_fn, expected_value, normalizer):
    scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
    labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
    expected_value = jnp.asarray(expected_value)

    mean_loss = loss_fn(scores, labels, reduce_fn=jnp.mean)
    sum_loss = loss_fn(scores, labels, reduce_fn=jnp.sum)

    np.testing.assert_allclose(
        mean_loss, jnp.sum(expected_value) / normalizer, rtol=1E-5)
    np.testing.assert_allclose(sum_loss, jnp.sum(expected_value))

  @parameterized.parameters([{
      "loss_fn": losses.softmax_loss,
      "expected_shape": (2,)
  }, {
      "loss_fn": losses.listmle_loss,
      "expected_shape": (2,)
  }, {
      "loss_fn": losses.poly1_softmax_loss,
      "expected_shape": (2,)
  }, {
      "loss_fn": losses.unique_softmax_loss,
      "expected_shape": (2,)
  }, {
      "loss_fn": losses.pairwise_hinge_loss,
      "expected_shape": (2, 9)
  }, {
      "loss_fn": losses.pairwise_logistic_loss,
      "expected_shape": (2, 9)
  }, {
      "loss_fn": losses.pairwise_soft_zero_one_loss,
      "expected_shape": (2, 9)
  }, {
      "loss_fn": losses.pairwise_mse_loss,
      "expected_shape": (2, 9)
  }, {
      "loss_fn": losses.pairwise_qr_loss,
      "expected_shape": (2, 9)
  }, {
      "loss_fn": losses.pointwise_sigmoid_loss,
      "expected_shape": (2, 3)
  }, {
      "loss_fn": losses.pointwise_mse_loss,
      "expected_shape": (2, 3)
  }])
  def test_computes_unreduced_loss(self, loss_fn, expected_shape):
    scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
    labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])

    none_loss = loss_fn(scores, labels, reduce_fn=None)
    sum_loss = loss_fn(scores, labels, reduce_fn=jnp.sum)

    self.assertEqual(none_loss.shape, expected_shape)
    self.assertEqual(jnp.sum(none_loss), sum_loss)

  @parameterized.parameters([
      losses.pointwise_mse_loss,
      losses.pointwise_sigmoid_loss,
      losses.pairwise_hinge_loss,
      losses.pairwise_logistic_loss,
      losses.pairwise_soft_zero_one_loss,
      losses.pairwise_mse_loss,
      losses.pairwise_qr_loss,
      losses.softmax_loss,
      losses.unique_softmax_loss,
  ])
  def test_computes_loss_value_with_segments(self, loss_fn):
    scores = jnp.asarray([0.0, 3.0, 1.0, 2.0, 4.0, 3.0, 0.0, 2.0])
    labels = jnp.asarray([0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0])
    segments = jnp.asarray([0, 0, 1, 1, 1, 2, 3, 3])

    list_scores = jnp.asarray(
        [[0.0, 3.0, 0.0], [1.0, 2.0, 4.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    )
    list_labels = jnp.asarray(
        [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    list_mask = jnp.asarray(
        [[1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=jnp.bool_
    )

    loss = loss_fn(scores, labels, segments=segments)
    expected_loss = loss_fn(
        list_scores, list_labels, where=list_mask
    )

    np.testing.assert_allclose(expected_loss, loss, rtol=1E-5)

  @parameterized.parameters([
      losses.pointwise_mse_loss,
      losses.pointwise_sigmoid_loss,
      losses.pairwise_hinge_loss,
      losses.pairwise_logistic_loss,
      losses.pairwise_soft_zero_one_loss,
      losses.pairwise_mse_loss,
      losses.pairwise_qr_loss,
      losses.softmax_loss,
      losses.unique_softmax_loss,
  ])
  def test_computes_loss_value_with_segments_and_mask(self, loss_fn):
    scores = jnp.asarray([0.0, 3.0, 1.0, 2.0, 4.0, 3.0, 0.0, 2.0, 0.0, 0.0])
    labels = jnp.asarray([0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    segments = jnp.asarray([0, 0, 1, 1, 1, 2, 3, 3, 0, 0])
    where = jnp.asarray([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=jnp.bool_)

    list_scores = jnp.asarray(
        [[0.0, 3.0, 0.0], [1.0, 2.0, 4.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    )
    list_labels = jnp.asarray(
        [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    list_mask = jnp.asarray(
        [[1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=jnp.bool_
    )

    loss = loss_fn(scores, labels, segments=segments, where=where)
    expected_loss = loss_fn(
        list_scores, list_labels, where=list_mask
    )

    np.testing.assert_allclose(expected_loss, loss, rtol=1E-5)

  @parameterized.parameters([
      losses.softmax_loss,
      losses.listmle_loss,
      losses.pairwise_hinge_loss,
      losses.pairwise_logistic_loss,
      losses.pairwise_soft_zero_one_loss,
      losses.pointwise_sigmoid_loss,
      losses.pointwise_mse_loss,
      losses.pairwise_mse_loss,
      losses.pairwise_qr_loss,
      losses.poly1_softmax_loss,
      losses.unique_softmax_loss,
  ])
  def test_computes_loss_value_with_where(self, loss_fn):
    scores = jnp.asarray([0., 3., 1., 2.])
    labels = jnp.asarray([0., 0., 2., 1.])
    where = jnp.asarray([True, True, True, False])
    expected_scores = jnp.asarray([0., 3., 1.])
    expected_labels = jnp.asarray([0., 0., 2.])

    loss = loss_fn(scores, labels, where=where)
    expected_loss = loss_fn(expected_scores, expected_labels)

    np.testing.assert_allclose(expected_loss, loss)

  @parameterized.parameters([
      losses.softmax_loss,
      losses.listmle_loss,
      losses.pairwise_hinge_loss,
      losses.pairwise_logistic_loss,
      losses.pairwise_soft_zero_one_loss,
      losses.pointwise_sigmoid_loss,
      losses.pointwise_mse_loss,
      losses.pairwise_mse_loss,
      losses.pairwise_qr_loss,
      losses.poly1_softmax_loss,
      losses.unique_softmax_loss,
  ])
  def test_computes_loss_value_with_all_masked(self, loss_fn):
    scores = jnp.asarray([0., 3., 1., 2.])
    labels = jnp.asarray([0., 0., 1., 1.])
    where = jnp.asarray([False, False, False, False])

    loss = loss_fn(scores, labels, where=where)

    np.testing.assert_allclose(jnp.asarray(0.), loss, atol=1E-7)

  @parameterized.parameters([
      losses.softmax_loss,
      losses.listmle_loss,
      losses.pairwise_hinge_loss,
      losses.pairwise_logistic_loss,
      losses.pairwise_soft_zero_one_loss,
      losses.pointwise_sigmoid_loss,
      losses.pointwise_mse_loss,
      losses.pairwise_mse_loss,
      losses.pairwise_qr_loss,
      losses.poly1_softmax_loss,
      losses.unique_softmax_loss,
  ])
  def test_computes_loss_with_arbitrary_batch_dimensions(self, loss_fn):
    scores = jnp.asarray([2., 3., 1.])
    labels = jnp.asarray([0., 0., 1.])
    where = jnp.asarray([False, True, True])
    original_loss = loss_fn(scores, labels, where=where)

    scores = jnp.asarray([[[[2., 3., 1.]]]])
    labels = jnp.asarray([[[[0., 0., 1.]]]])
    where = jnp.asarray([[[[False, True, True]]]])
    batched_loss = loss_fn(scores, labels, where=where)

    np.testing.assert_allclose(original_loss, batched_loss)

  @parameterized.parameters([
      losses.softmax_loss,
      losses.listmle_loss,
      losses.pairwise_hinge_loss,
      losses.pairwise_logistic_loss,
      losses.pairwise_soft_zero_one_loss,
      losses.pointwise_sigmoid_loss,
      losses.pointwise_mse_loss,
      losses.pairwise_mse_loss,
      losses.pairwise_qr_loss,
      losses.poly1_softmax_loss,
      losses.unique_softmax_loss,
  ])
  def test_grad_does_not_return_nan_for_zero_labels(self, loss_fn):
    scores = jnp.asarray([0., 3., 1., 2.])
    labels = jnp.asarray([0., 0., 0., 0.])

    grads = jax.grad(loss_fn)(scores, labels, reduce_fn=jnp.mean)

    np.testing.assert_array_equal(
        jnp.isnan(grads), jnp.zeros_like(jnp.isnan(grads)))

  @parameterized.parameters([
      losses.softmax_loss,
      losses.listmle_loss,
      losses.pairwise_hinge_loss,
      losses.pairwise_logistic_loss,
      losses.pairwise_soft_zero_one_loss,
      losses.pointwise_sigmoid_loss,
      losses.pointwise_mse_loss,
      losses.pairwise_mse_loss,
      losses.pairwise_qr_loss,
      losses.poly1_softmax_loss,
      losses.unique_softmax_loss,
  ])
  def test_grad_does_not_return_nan_with_all_masked(self, loss_fn):
    scores = jnp.asarray([0., 3., 1., 2.])
    labels = jnp.asarray([0., 0., 1., 1.])
    where = jnp.asarray([False, False, False, False])

    grads = jax.grad(loss_fn)(scores, labels, where=where, reduce_fn=jnp.mean)

    np.testing.assert_array_equal(
        jnp.isnan(grads), jnp.zeros_like(jnp.isnan(grads)))

  @parameterized.parameters([
      losses.softmax_loss,
      losses.listmle_loss,
      losses.pairwise_hinge_loss,
      losses.pairwise_logistic_loss,
      losses.pairwise_soft_zero_one_loss,
      losses.pointwise_sigmoid_loss,
      losses.pointwise_mse_loss,
      losses.pairwise_mse_loss,
      losses.pairwise_qr_loss,
      losses.poly1_softmax_loss,
      losses.unique_softmax_loss,
  ])
  def test_ignores_lists_containing_only_invalid_items(self, loss_fn):
    scores = jnp.asarray([[0.0, 3.0, 1.0, 2.0], [3.0, 1.0, 4.0, 2.0]])
    labels = jnp.asarray([[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 0.0]])
    mask = jnp.asarray([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=jnp.bool_)

    output = loss_fn(scores, labels, where=mask)
    expected = loss_fn(scores[0, :], labels[0, :])

    np.testing.assert_allclose(output, expected)


def load_tests(loader, tests, ignore):
  del loader, ignore  # Unused.
  tests.addTests(
      doctest.DocTestSuite(losses, globs={
          "jax": jax,
          "jnp": jnp,
          "rax": rax
      }))
  return tests


if __name__ == "__main__":
  absltest.main()
