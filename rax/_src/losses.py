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

import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from rax._src import utils
from rax._src.protocols import ReduceFn


def softmax_loss(
    scores: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    mask: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
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
  >>> mask = jnp.array([[True, True, False], [True, True, True]])
  >>> rax.softmax_loss(scores, labels, mask=mask, reduce_fn=jnp.mean)
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
    mask: An optional [..., list_size]-jnp.ndarray, indicating which items are
      valid for computing the loss. Items for which the mask is False will be
      ignored when computing the loss.
    weights: An optional [..., list_size]-jnp.ndarray, indicating the weight for
      each item.
    reduce_fn: An optional Callable that reduces the loss values. The callable
      should accept a loss tensor and an optional `where` tensor indicating
      which elements to include in the reduction. Can be `jnp.sum` or
      `jnp.mean`. If `None`, no reduction is performed.

  Returns:
    The softmax loss.
  """
  # Applies mask so that masked elements do not count towards the loss.
  if mask is not None:
    labels = jnp.where(mask, labels, jnp.zeros_like(labels))
    scores = jnp.where(mask, scores, -jnp.ones_like(scores) * jnp.inf)

  # Apply weights to labels.
  if weights is not None:
    labels *= weights

  # Scales labels and scores to match the cross entropy loss.
  labels_probabilities = utils.normalize_probabilities(labels, mask, axis=-1)
  scores_log_softmax = jax.nn.log_softmax(scores, axis=-1)

  # Computes per-element cross entropy.
  softmax_cross_entropy = labels_probabilities * scores_log_softmax

  # Reduces softmax cross-entropy loss.
  loss = -jnp.sum(softmax_cross_entropy, axis=-1, where=mask)
  return utils.safe_reduce(loss, reduce_fn=reduce_fn)


def compute_pairwise_loss(
    scores: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    mask: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
    reduce_fn: ReduceFn = jnp.sum,
    pair_loss_fn: Callable[[jnp.float_, jnp.ndarray], jnp.float_],
) -> jnp.ndarray:
  """Computes a pairwise loss.

  Args:
    scores: A [..., list_size]-jnp.ndarray, indicating the score of each item.
    labels: A [..., list_size]-jnp.ndarray, indicating the relevance label for
      each item.
    mask: An optional [..., list_size]-jnp.ndarray, indicating which items are
      valid for computing the loss. Items for which the mask is False will be
      ignored when computing the loss.
    weights: An optional [..., list_size]-jnp.ndarray, indicating the weight for
      each item.
    reduce_fn: An optional Callable that reduces the loss values. The callable
      should accept a loss tensor and an optional `where` tensor indicating
      which elements to include in the reduction. Can be `jnp.sum` or
      `jnp.mean`. If `None`, no reduction is performed.
    pair_loss_fn: A callable that computes the loss on pairs of scores.

  Returns:
    The reduced result of the given pairwise loss function.
  """
  label_i = jnp.expand_dims(labels, axis=-1)
  label_j = jnp.expand_dims(labels, axis=-2)
  valid_pairs = label_i > label_j

  score_i = jnp.broadcast_to(
      jnp.expand_dims(scores, axis=-1), valid_pairs.shape)
  score_j = jnp.broadcast_to(
      jnp.expand_dims(scores, axis=-2), valid_pairs.shape)
  vmap_last_axis = functools.partial(jax.vmap, in_axes=-1, out_axes=-1)
  loss_pairs = vmap_last_axis(vmap_last_axis(pair_loss_fn))(score_i, score_j)

  if mask is not None:
    mask_i = jnp.expand_dims(mask, axis=-1)
    mask_j = jnp.expand_dims(mask, axis=-2)
    valid_pairs &= mask_i & mask_j

  if weights is not None:
    loss_pairs *= weights[:, None]

  return utils.safe_reduce(loss_pairs, where=valid_pairs, reduce_fn=reduce_fn)


def pairwise_hinge_loss(
    scores: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    mask: Optional[jnp.ndarray] = None,
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
  >>> mask = jnp.array([[True, True, False], [True, True, True]])
  >>> rax.pairwise_hinge_loss(
  ...     scores, labels, mask=mask, reduce_fn=jnp.mean)
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
    mask: An optional [..., list_size]-jnp.ndarray, indicating which items are
      valid for computing the loss. Items for which the mask is False will be
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

  def _hinge_loss(score_i: float, score_j: float) -> float:
    return jax.nn.relu(1. - (score_i - score_j))

  return compute_pairwise_loss(
      scores,
      labels,
      mask=mask,
      weights=weights,
      reduce_fn=reduce_fn,
      pair_loss_fn=_hinge_loss)


def pairwise_logistic_loss(
    scores: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    mask: Optional[jnp.ndarray] = None,
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
  >>> mask = jnp.array([[True, True, False], [True, True, True]])
  >>> rax.pairwise_logistic_loss(
  ...     scores, labels, mask=mask, reduce_fn=jnp.mean)
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
    mask: An optional [..., list_size]-jnp.ndarray, indicating which items are
      valid for computing the loss. Items for which the mask is False will be
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

  def _logistic_loss(score_i: float, score_j: float) -> float:
    score_diff = score_i - score_j
    return jax.nn.relu(-score_diff) + jnp.log1p(jnp.exp(-jnp.abs(score_diff)))

  return compute_pairwise_loss(
      scores,
      labels,
      mask=mask,
      weights=weights,
      reduce_fn=reduce_fn,
      pair_loss_fn=_logistic_loss)


def compute_pointwise_loss(
    scores: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    mask: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
    reduce_fn: ReduceFn = jnp.sum,
    point_loss_fn: Callable[[jnp.float_, jnp.float_],
                            jnp.float_]) -> jnp.ndarray:
  """Computes a pointwise loss.

  Args:
    scores: A [..., list_size]-jnp.ndarray, indicating the score of each item.
    labels: A [..., list_size]-jnp.ndarray, indicating the relevance label for
      each item.
    mask: An optional [..., list_size]-jnp.ndarray, indicating which items are
      valid for computing the loss. Items for which the mask is False will be
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

  if mask is not None:
    results *= mask
    valid_items = jnp.int32(mask)

  if weights is not None:
    results *= weights

  return utils.safe_reduce(results, where=valid_items, reduce_fn=reduce_fn)


def sigmoid_cross_entropy_loss(
    scores: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    mask: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
    reduce_fn: ReduceFn = jnp.sum) -> jnp.ndarray:
  r"""Ranking sigmoid cross entropy loss.

  Standalone usage:

  >>> scores = jnp.array([2., 1., 3.])
  >>> labels = jnp.array([1., 0., 0.])
  >>> rax.sigmoid_cross_entropy_loss(scores, labels)
  DeviceArray(4.488777, dtype=float32)

  Usage with mean reduction across a batch and a mask to indicate valid items:

  >>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> mask = jnp.array([[True, True, False], [True, True, True]])
  >>> rax.sigmoid_cross_entropy_loss(
  ...     scores, labels, mask=mask, reduce_fn=jnp.mean)
  DeviceArray(0.78578836, dtype=float32)

  Usage with `vmap` batching and a mask to indicate valid items:

  >>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> jax.vmap(rax.sigmoid_cross_entropy_loss)(scores, labels)
  DeviceArray([4.488777, 2.488752], dtype=float32)

  Definition:

  .. math::
      \operatorname{sigmoid_cross_entropy_loss}(s, y) =
      \sum_i y_i * -log(sigmoid(s_i)) + (1 - y_i) * -log(1 - sigmoid(s_i))

  This loss converts graded relevance to binary relevance by considering items
  with `label >= 1` as relevant and items with `label < 1` as non-relevant.

  Args:
    scores: A [..., list_size]-jnp.ndarray, indicating the score of each item.
    labels: A [..., list_size]-jnp.ndarray, indicating the relevance label for
      each item.
    mask: An optional [..., list_size]-jnp.ndarray, indicating which items are
      valid for computing the loss. Items for which the mask is False will be
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
  def _cross_entropy_loss(score: float, label: float) -> float:
    return (jax.nn.relu(score) - score * label +
            jnp.log(1. + jnp.exp(-jnp.abs(score))))

  return compute_pointwise_loss(
      scores,
      labels,
      mask=mask,
      weights=weights,
      reduce_fn=reduce_fn,
      point_loss_fn=_cross_entropy_loss)


def mse_loss(scores: jnp.ndarray,
             labels: jnp.ndarray,
             *,
             mask: Optional[jnp.ndarray] = None,
             weights: Optional[jnp.ndarray] = None,
             reduce_fn: ReduceFn = jnp.sum) -> jnp.ndarray:
  r"""Ranking mean squared error loss.

  Standalone usage:

  >>> scores = jnp.array([2., 1., 3.])
  >>> labels = jnp.array([1., 0., 0.])
  >>> rax.mse_loss(scores, labels)
  DeviceArray(11., dtype=float32)

  Usage with mean reduction across a batch and a mask to indicate valid items:

  >>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> mask = jnp.array([[True, True, False], [True, True, True]])
  >>> rax.mse_loss(scores, labels, mask=mask, reduce_fn=jnp.mean)
  DeviceArray(0.7, dtype=float32)

  Usage with `vmap` batching and a mask to indicate valid items:

  >>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> jax.vmap(rax.mse_loss)(scores, labels)
  DeviceArray([11. ,  1.5], dtype=float32)

  Definition:

  .. math::
      \operatorname{mse_loss}(s, y) = \sum_i (y_i - s_i)^2

  Args:
    scores: A [..., list_size]-jnp.ndarray, indicating the score of each item.
    labels: A [..., list_size]-jnp.ndarray, indicating the relevance label for
      each item.
    mask: An optional [..., list_size]-jnp.ndarray, indicating which items are
      valid for computing the loss. Items for which the mask is False will be
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

  def _mse_loss(score: float, label: float) -> float:
    return (score - label)**2.0

  return compute_pointwise_loss(
      scores,
      labels,
      mask=mask,
      weights=weights,
      reduce_fn=reduce_fn,
      point_loss_fn=_mse_loss)


def pairwise_mse_loss(
    scores: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    mask: Optional[jnp.ndarray] = None,
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
  >>> mask = jnp.array([[True, True, False], [True, True, True]])
  >>> rax.pairwise_mse_loss(
  ...     scores, labels, mask=mask, reduce_fn=jnp.mean)
  DeviceArray(0.07692308, dtype=float32)

  Usage with `vmap` batching and a mask to indicate valid items:

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
    mask: An optional [..., list_size]-jnp.ndarray, indicating which items are
      valid for computing the loss. Items for which the mask is False will be
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
  # Construct (score_i - score_j)
  scores = jnp.expand_dims(scores, axis=-1) - jnp.expand_dims(scores, axis=-2)
  scores = jnp.reshape(scores, scores.shape[:-2] + (-1,))

  # Construct (label_i - label_j)
  labels = jnp.expand_dims(labels, axis=-1) - jnp.expand_dims(labels, axis=-2)
  labels = jnp.reshape(labels, labels.shape[:-2] + (-1,))

  # Construct (mask_i & mask_j)
  if mask is not None:
    mask = jnp.expand_dims(mask, axis=-1) & jnp.expand_dims(mask, axis=-2)
    mask = jnp.reshape(mask, mask.shape[:-2] + (-1,))

  # Construct weights_i (broadcasted to pairwise matrix).
  if weights is not None:
    weights = jnp.expand_dims(weights, axis=-1)
    weights = jnp.repeat(weights, axis=-1, repeats=weights.shape[-2])
    weights = jnp.reshape(weights, weights.shape[:-2] + (-1,))

  # Compute mse loss.
  return mse_loss(
      scores, labels, mask=mask, weights=weights, reduce_fn=reduce_fn)
