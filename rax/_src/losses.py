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

"""Ranking losses in JAX.

A ranking loss function is a callable that accepts a scores tensor, a labels
tensor, and, optionally a mask tensor. In addition to these arguments, loss
functions *may* accept additional optional keyword arguments, e.g. for weights,
however this depends on the specific loss function and is not required.

The loss functions operate on the last dimension of its inputs. The leading
dimensions are considered batch dimensions. To compute per-list losses, for
example to apply per-list weighting or for distributed computing of losses
across devices, please use standard JAX transformations such as `jax.vmap` or
`jax.pmap`.

To compute gradients of each loss function, please use standard JAX
transformations such as `jax.grad` or `jax.value_and_grad`.

Example usage:

>>> scores = jnp.asarray([[0., 1., 3.], [1., 2., 0.]])
>>> labels = jnp.asarray([[0., 0., 1.], [1., 0., 0.]])
>>> rax.softmax_loss(scores, labels, reduce_fn=jnp.mean)
DeviceArray(0.788726, dtype=float32)
>>> jax.grad(rax.softmax_loss)(scores, labels, reduce_fn=jnp.mean)
DeviceArray([[ 0.02100503,  0.0570976 , -0.07810265],
             [-0.37763578,  0.33262047,  0.04501529]], dtype=float32)

"""

import operator
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from rax._src import utils
from rax._src.protocols import ReduceFn


def softmax_loss(scores: jnp.ndarray,
                 labels: jnp.ndarray,
                 *,
                 where: Optional[jnp.ndarray] = None,
                 weights: Optional[jnp.ndarray] = None,
                 label_fn: Callable[..., jnp.ndarray] = lambda a, where: a,
                 reduce_fn: Optional[ReduceFn] = jnp.sum) -> jnp.float_:
  r"""Ranking softmax loss.

  Standalone usage:

  >>> scores = jnp.array([2., 1., 3.])
  >>> labels = jnp.array([1., 0., 0.])
  >>> rax.softmax_loss(scores, labels)
  DeviceArray(1.4076059, dtype=float32)

  Usage with mean reduction across a batch and a mask to indicate valid items:

  >>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> where = jnp.array([[True, True, False], [True, True, True]])
  >>> rax.softmax_loss(scores, labels, where=where, reduce_fn=jnp.mean)
  DeviceArray(0.49676564, dtype=float32)

  Usage with `vmap` batching:

  >>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> jax.vmap(rax.softmax_loss)(scores, labels)
  DeviceArray([1.4076059 , 0.68026966], dtype=float32)

  Definition:

  .. math::
      \operatorname{softmax_loss}(s, y) =
      \sum_i \frac{y_i}{\sum_j y_j} \log\operatorname{softmax}(s)_i

  Args:
    scores: A [..., list_size]-jnp.ndarray, indicating the score of each item.
    labels: A [..., list_size]-jnp.ndarray, indicating the relevance label for
      each item.
    where: An optional [..., list_size]-jnp.ndarray, indicating which items are
      valid for computing the loss. Items for which this is False will be
      ignored when computing the loss.
    weights: An optional [..., list_size]-jnp.ndarray, indicating the weight for
      each item.
    label_fn: A label function that maps labels to probabilities. Default keeps
      labels as-is.
    reduce_fn: An optional Callable that reduces the loss values. The callable
      should accept a loss tensor and an optional `where` tensor indicating
      which elements to include in the reduction. Can be `jnp.sum` or
      `jnp.mean`. If `None`, no reduction is performed.

  Returns:
    The softmax loss.
  """
  # Applies where so that whereed elements do not count towards the loss.
  if where is not None:
    labels = jnp.where(where, labels, jnp.zeros_like(labels))
    scores = jnp.where(where, scores, -jnp.ones_like(scores) * jnp.inf)

  # Apply weights to labels.
  if weights is not None:
    labels *= weights

  # Scales labels and scores to match the cross entropy loss.
  labels_probabilities = label_fn(labels, where=where)
  scores_log_softmax = jax.nn.log_softmax(scores, axis=-1)

  # Computes per-element cross entropy.
  softmax_cross_entropy = labels_probabilities * scores_log_softmax

  # Reduces softmax cross-entropy loss.
  loss = -jnp.sum(softmax_cross_entropy, axis=-1, where=where)
  return utils.safe_reduce(loss, reduce_fn=reduce_fn)


def compute_pairs(
    a: jnp.ndarray, op: Callable[[jnp.ndarray, jnp.ndarray],
                                 jnp.ndarray]) -> jnp.ndarray:
  """Computes pairs based on values of `a` and the given pairwise `op`.

  Args:
    a: The array used to form pairs. The last axis is used to form pairs.
    op: The binary op to map a pair of values to a single value.

  Returns:
    A new array with the same leading dimensions as `a`, but with the last
    dimension expanded so it includes all pairs `op(a[..., i], a[..., j])`
  """
  a_i = jnp.expand_dims(a, -1)
  a_j = jnp.expand_dims(a, -2)
  result_shape = jnp.broadcast_shapes(a_i.shape, a_j.shape)
  result = jnp.broadcast_to(op(a_i, a_j), result_shape)
  out_shape = tuple(result.shape[:-2]) + (result.shape[-2] * result.shape[-1],)
  return jnp.reshape(result, out_shape)


def pairwise_hinge_loss(scores: jnp.ndarray,
                        labels: jnp.ndarray,
                        *,
                        where: Optional[jnp.ndarray] = None,
                        weights: Optional[jnp.ndarray] = None,
                        reduce_fn: ReduceFn = jnp.sum) -> jnp.ndarray:
  r"""Ranking pairwise hinge loss.

  Standalone usage:

  >>> scores = jnp.array([2., 1., 3.])
  >>> labels = jnp.array([1., 0., 0.])
  >>> rax.pairwise_hinge_loss(scores, labels)
  DeviceArray(2., dtype=float32)

  Usage with mean reduction across a batch and a mask to indicate valid items:

  >>> scores = jnp.array([[2., 1., 0.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> where = jnp.array([[True, True, False], [True, True, True]])
  >>> rax.pairwise_hinge_loss(
  ...     scores, labels, where=where, reduce_fn=jnp.mean)
  DeviceArray(0.16666667, dtype=float32)

  Usage with `vmap` batching:

  >>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> jax.vmap(rax.pairwise_hinge_loss)(scores, labels)
  DeviceArray([2. , 0.5], dtype=float32)

  Definition:

  .. math::
      \operatorname{hinge_loss}(s, y) =
      \sum_i \sum_j I[y_i > y_j] \max(0, 1 - (s_i - s_j)

  Args:
    scores: A [..., list_size]-jnp.ndarray, indicating the score of each item.
    labels: A [..., list_size]-jnp.ndarray, indicating the relevance label for
      each item.
    where: An optional [..., list_size]-jnp.ndarray, indicating which items are
      valid for computing the loss. Items for which this is False will be
      ignored when computing the loss.
    weights: An optional [..., list_size]-jnp.ndarray, indicating the weight for
      each item.
    reduce_fn: An optional Callable that reduces the loss values. The callable
      should accept a loss tensor and an optional `where` tensor indicating
      which elements to include in the reduction. Can be `jnp.sum` or
      `jnp.mean`. If `None`, no reduction is performed.

  Returns:
    The pairwise hinge loss.
  """
  # Expand scores and labels into pairwise versions.
  scores = compute_pairs(scores, operator.sub)
  labels = compute_pairs(labels, operator.gt)

  # Compute hinge function on scores.
  scores = jax.nn.relu(1. - scores)

  # Apply mask to labels, so only valid items are considered.
  if where is not None:
    where = compute_pairs(where, operator.and_)
    labels &= where

  # Apply weights to scores.
  if weights is not None:
    weights = compute_pairs(weights, lambda x, y: x)
    scores *= weights

  return utils.safe_reduce(scores, where=labels, reduce_fn=reduce_fn)


def pairwise_logistic_loss(scores: jnp.ndarray,
                           labels: jnp.ndarray,
                           *,
                           where: Optional[jnp.ndarray] = None,
                           weights: Optional[jnp.ndarray] = None,
                           reduce_fn: ReduceFn = jnp.sum) -> jnp.ndarray:
  r"""Ranking pairwise logistic loss.

  Standalone usage:

  >>> scores = jnp.array([2., 1., 3.])
  >>> labels = jnp.array([1., 0., 0.])
  >>> rax.pairwise_logistic_loss(scores, labels)
  DeviceArray(1.6265235, dtype=float32)

  Usage with mean reduction across a batch and a mask to indicate valid items:

  >>> scores = jnp.array([[2., 1., 0.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> where = jnp.array([[True, True, False], [True, True, True]])
  >>> rax.pairwise_logistic_loss(
  ...     scores, labels, where=where, reduce_fn=jnp.mean)
  DeviceArray(0.3668668, dtype=float32)

  Usage with `vmap` batching:

  >>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> jax.vmap(rax.pairwise_logistic_loss)(scores, labels)
  DeviceArray([1.6265235, 0.7873387], dtype=float32)

  Definition:

  .. math::
      \operatorname{logistic_loss}(s, y) =
      \sum_i \sum_j I[y_i > y_j] \log(1 + \exp(-(s_i - s_j)))

  Args:
    scores: A [..., list_size]-jnp.ndarray, indicating the score of each item.
    labels: A [..., list_size]-jnp.ndarray, indicating the relevance label for
      each item.
    where: An optional [..., list_size]-jnp.ndarray, indicating which items are
      valid for computing the loss. Items for which this is False will be
      ignored when computing the loss.
    weights: An optional [..., list_size]-jnp.ndarray, indicating the weight for
      each item.
    reduce_fn: An optional Callable that reduces the loss values. The callable
      should accept a loss tensor and an optional `where` tensor indicating
      which elements to include in the reduction. Can be `jnp.sum` or
      `jnp.mean`. If `None`, no reduction is performed.

  Returns:
    The pairwise logistic loss.
  """
  # Expand scores and labels into pairwise versions.
  scores = compute_pairs(scores, operator.sub)
  labels = compute_pairs(labels, operator.gt)

  # Compute numerically stable version of logistic function on scores.
  scores = jax.nn.relu(-scores) + jnp.log1p(jnp.exp(-jnp.abs(scores)))

  # Apply mask to labels, so only valid items are considered.
  if where is not None:
    where = compute_pairs(where, operator.and_)
    labels &= where

  # Apply weights to scores.
  if weights is not None:
    weights = compute_pairs(weights, lambda x, y: x)
    scores *= weights

  return utils.safe_reduce(scores, where=labels, reduce_fn=reduce_fn)


def compute_pointwise_loss(
    scores: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    where: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
    reduce_fn: ReduceFn = jnp.sum,
    point_loss_fn: Callable[[jnp.float_, jnp.float_],
                            jnp.float_]) -> jnp.ndarray:
  """Computes a pointwise loss.

  Args:
    scores: A [..., list_size]-jnp.ndarray, indicating the score of each item.
    labels: A [..., list_size]-jnp.ndarray, indicating the relevance label for
      each item.
    where: An optional [..., list_size]-jnp.ndarray, indicating which items are
      valid for computing the loss. Items for which this is False will be
      ignored when computing the loss.
    weights: An optional [..., list_size]-jnp.ndarray, indicating the weight for
      each item.
    reduce_fn: An optional Callable that reduces the loss values. The callable
      should accept a loss tensor and an optional `where` tensor indicating
      which elements to include in the reduction. Can be `jnp.sum` or
      `jnp.mean`. If `None`, no reduction is performed.
    point_loss_fn: A callable that computes the loss on individual items.

  Returns:
    The reduced result of the given pointwise loss function.
  """
  results = jax.vmap(point_loss_fn, in_axes=-1, out_axes=-1)(scores, labels)
  valid_items = jnp.ones_like(labels, dtype=jnp.int32)

  if where is not None:
    results *= where
    valid_items = jnp.int32(where)

  if weights is not None:
    results *= weights

  return utils.safe_reduce(results, where=valid_items, reduce_fn=reduce_fn)


def pointwise_sigmoid_loss(scores: jnp.ndarray,
                           labels: jnp.ndarray,
                           *,
                           where: Optional[jnp.ndarray] = None,
                           weights: Optional[jnp.ndarray] = None,
                           reduce_fn: ReduceFn = jnp.sum) -> jnp.ndarray:
  r"""Ranking sigmoid cross entropy loss.

  Standalone usage:

  >>> scores = jnp.array([2., 1., 3.])
  >>> labels = jnp.array([1., 0., 0.])
  >>> rax.pointwise_sigmoid_loss(scores, labels)
  DeviceArray(4.488777, dtype=float32)

  Usage with mean reduction across a batch and a mask to indicate valid items:

  >>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> where = jnp.array([[True, True, False], [True, True, True]])
  >>> rax.pointwise_sigmoid_loss(
  ...     scores, labels, where=where, reduce_fn=jnp.mean)
  DeviceArray(0.78578836, dtype=float32)

  Usage with `vmap` batching:

  >>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> jax.vmap(rax.pointwise_sigmoid_loss)(scores, labels)
  DeviceArray([4.488777, 2.488752], dtype=float32)

  Definition:

  .. math::
      \operatorname{pointwise_sigmoid_loss}(s, y) =
      \sum_i y_i * -log(sigmoid(s_i)) + (1 - y_i) * -log(1 - sigmoid(s_i))

  This loss converts graded relevance to binary relevance by considering items
  with `label >= 1` as relevant and items with `label < 1` as non-relevant.

  Args:
    scores: A [..., list_size]-jnp.ndarray, indicating the score of each item.
    labels: A [..., list_size]-jnp.ndarray, indicating the relevance label for
      each item.
    where: An optional [..., list_size]-jnp.ndarray, indicating which items are
      valid for computing the loss. Items for which this is False will be
      ignored when computing the loss.
    weights: An optional [..., list_size]-jnp.ndarray, indicating the weight for
      each item.
    reduce_fn: An optional Callable that reduces the loss values. The callable
      should accept a loss tensor and an optional `where` tensor indicating
      which elements to include in the reduction. Can be `jnp.sum` or
      `jnp.mean`. If `None`, no reduction is performed.

  Returns:
    The sigmoid cross entropy loss.
  """

  # Convert to binary relevance labels.
  labels = jnp.where(labels >= 1, jnp.ones_like(labels), jnp.zeros_like(labels))

  # A numerically stable version of sigmoid cross entropy.
  loss = (
      jax.nn.relu(scores) - scores * labels +
      jnp.log(1. + jnp.exp(-jnp.abs(scores))))

  if weights is not None:
    loss *= weights

  return utils.safe_reduce(loss, where=where, reduce_fn=reduce_fn)


def pointwise_mse_loss(scores: jnp.ndarray,
                       labels: jnp.ndarray,
                       *,
                       where: Optional[jnp.ndarray] = None,
                       weights: Optional[jnp.ndarray] = None,
                       reduce_fn: ReduceFn = jnp.sum) -> jnp.ndarray:
  r"""Ranking mean squared error loss.

  Standalone usage:

  >>> scores = jnp.array([2., 1., 3.])
  >>> labels = jnp.array([1., 0., 0.])
  >>> rax.pointwise_mse_loss(scores, labels)
  DeviceArray(11., dtype=float32)

  Usage with mean reduction across a batch and a mask to indicate valid items:

  >>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> where = jnp.array([[True, True, False], [True, True, True]])
  >>> rax.pointwise_mse_loss(scores, labels, where=where, reduce_fn=jnp.mean)
  DeviceArray(0.7, dtype=float32)

  Usage with `vmap` batching:

  >>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> jax.vmap(rax.pointwise_mse_loss)(scores, labels)
  DeviceArray([11. ,  1.5], dtype=float32)

  Definition:

  .. math::
      \operatorname{pointwise_mse_loss}(s, y) = \sum_i (y_i - s_i)^2

  Args:
    scores: A [..., list_size]-jnp.ndarray, indicating the score of each item.
    labels: A [..., list_size]-jnp.ndarray, indicating the relevance label for
      each item.
    where: An optional [..., list_size]-jnp.ndarray, indicating which items are
      valid for computing the loss. Items for which this is False will be
      ignored when computing the loss.
    weights: An optional [..., list_size]-jnp.ndarray, indicating the weight for
      each item.
    reduce_fn: An optional Callable that reduces the loss values. The callable
      should accept a loss tensor and an optional `where` tensor indicating
      which elements to include in the reduction. Can be `jnp.sum` or
      `jnp.mean`. If `None`, no reduction is performed.

  Returns:
    The mean squared error loss.
  """
  loss = jnp.square(scores - labels)

  if weights is not None:
    loss *= weights

  return utils.safe_reduce(loss, where=where, reduce_fn=reduce_fn)


def pairwise_mse_loss(scores: jnp.ndarray,
                      labels: jnp.ndarray,
                      *,
                      where: Optional[jnp.ndarray] = None,
                      weights: Optional[jnp.ndarray] = None,
                      reduce_fn: ReduceFn = jnp.sum) -> jnp.ndarray:
  r"""Pairwise mean squared error loss.

  Standalone usage:

  >>> scores = jnp.array([2., 1., 3.])
  >>> labels = jnp.array([1., 0., 0.])
  >>> rax.pairwise_mse_loss(scores, labels)
  DeviceArray(16., dtype=float32)

  Usage with mean reduction across a batch and a mask to indicate valid items:

  >>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> where = jnp.array([[True, True, False], [True, True, True]])
  >>> rax.pairwise_mse_loss(
  ...     scores, labels, where=where, reduce_fn=jnp.mean)
  DeviceArray(0.07692308, dtype=float32)

  Usage with `vmap` batching:

  >>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> jax.vmap(rax.pairwise_mse_loss)(scores, labels)
  DeviceArray([16.,  1.], dtype=float32)

  Definition:

  .. math::
      \operatorname{pairwise_mse_loss}(s, y) =
      \sum_i \sum_j ((y_i - y_j) - (s_i - s_j))^2

  Args:
    scores: A [..., list_size]-jnp.ndarray, indicating the score of each item.
    labels: A [..., list_size]-jnp.ndarray, indicating the relevance label for
      each item.
    where: An optional [..., list_size]-jnp.ndarray, indicating which items are
      valid for computing the loss. Items for which this is False will be
      ignored when computing the loss.
    weights: An optional [..., list_size]-jnp.ndarray, indicating the weight for
      each item.
    reduce_fn: An optional Callable that reduces the loss values. The callable
      should accept a loss tensor and an optional `where` tensor indicating
      which elements to include in the reduction. Can be `jnp.sum` or
      `jnp.mean`. If `None`, no reduction is performed.

  Returns:
    The pairwise mean squared error loss.
  """
  # Expand scores and labels into pairwise versions.
  scores = compute_pairs(scores, operator.sub)
  labels = compute_pairs(labels, operator.sub)

  # Compute pairwise mask.
  if where is not None:
    where = compute_pairs(where, operator.and_)

  # Compute squared error between score pairs and label pairs.
  loss = jnp.square(scores - labels)

  # Apply weights to scores.
  if weights is not None:
    weights = compute_pairs(weights, lambda x, y: x)
    loss *= weights

  return utils.safe_reduce(loss, where=where, reduce_fn=reduce_fn)
