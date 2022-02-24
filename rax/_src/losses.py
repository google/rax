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

"""Implementations of common ranking losses in JAX.

A ranking loss is a differentiable function that expresses the cost of a ranking
induced by item scores compared to a ranking induced from relevance labels. Rax
provides a number of ranking losses as JAX functions that are implemented
according to the :class:`~rax.types.LossFn` interface.

Loss functions are designed to operate on the last dimension of its inputs. The
leading dimensions are considered batch dimensions. To compute per-list losses,
for example to apply per-list weighting or for distributed computing of losses
across devices, please use standard JAX transformations such as :func:`jax.vmap`
or :func:`jax.pmap`.

Standalone usage:

>>> scores = jnp.array([2., 1., 3.])
>>> labels = jnp.array([1., 0., 0.])
>>> rax.softmax_loss(scores, labels)
DeviceArray(1.4076059, dtype=float32)

Usage with a batch of data and a mask to indicate valid items.

>>> scores = jnp.array([[2., 1., 0.], [1., 0.5, 1.5]])
>>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
>>> where = jnp.array([[True, True, False], [True, True, True]])
>>> rax.pairwise_hinge_loss(
...     scores, labels, where=where, reduce_fn=jnp.mean)
DeviceArray(0.16666667, dtype=float32)

To compute gradients of each loss function, please use standard JAX
transformations such as :func:`jax.grad` or :func:`jax.value_and_grad`:

>>> scores = jnp.asarray([[0., 1., 3.], [1., 2., 0.]])
>>> labels = jnp.asarray([[0., 0., 1.], [1., 0., 0.]])
>>> jax.grad(rax.softmax_loss)(scores, labels, reduce_fn=jnp.mean)
DeviceArray([[ 0.02100503,  0.0570976 , -0.07810265],
             [-0.37763578,  0.33262047,  0.04501529]], dtype=float32)

"""

import operator
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from rax._src import utils
from rax._src.types import Array
from rax._src.types import ReduceFn


def softmax_loss(scores: Array,
                 labels: Array,
                 *,
                 where: Optional[Array] = None,
                 weights: Optional[Array] = None,
                 label_fn: Callable[..., Array] = lambda a, where: a,
                 reduce_fn: Optional[ReduceFn] = jnp.sum) -> Array:
  r"""Softmax loss.

  Definition:

  .. math::
      \ell(s, y) =
      \sum_i y_i \log \frac{\exp(s_i)}{\sum_j \exp(s_j)}

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the loss. Items for which
      this is False will be ignored when computing the loss.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the weight for each item.
    label_fn: A label function that maps labels to probabilities. Default keeps
      labels as-is.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

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


def compute_pairs(a: Array, op: Callable[[Array, Array], Array]) -> Array:
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


def pairwise_hinge_loss(scores: Array,
                        labels: Array,
                        *,
                        where: Optional[Array] = None,
                        weights: Optional[Array] = None,
                        reduce_fn: ReduceFn = jnp.sum) -> Array:
  r"""Pairwise hinge loss.

  Definition:

  .. math::
      \ell(s, y) =
      \sum_i \sum_j I[y_i > y_j] \max(0, 1 - (s_i - s_j))

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the loss. Items for which
      this is False will be ignored when computing the loss.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the weight for each item.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

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


def pairwise_logistic_loss(scores: Array,
                           labels: Array,
                           *,
                           where: Optional[Array] = None,
                           weights: Optional[Array] = None,
                           reduce_fn: ReduceFn = jnp.sum) -> Array:
  r"""Pairwise logistic loss.

  Definition :cite:p:`burges2005learning`:

  .. math::
      \ell(s, y) =
      \sum_i \sum_j I[y_i > y_j] \log(1 + \exp(-(s_i - s_j)))

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the loss. Items for which
      this is False will be ignored when computing the loss.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the weight for each item.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

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


def pointwise_sigmoid_loss(scores: Array,
                           labels: Array,
                           *,
                           where: Optional[Array] = None,
                           weights: Optional[Array] = None,
                           reduce_fn: ReduceFn = jnp.sum) -> Array:
  r"""Sigmoid cross entropy loss.

  Definition:

  .. math::
      \ell(s, y) =
      \sum_i y_i * -log(sigmoid(s_i)) + (1 - y_i) * -log(1 - sigmoid(s_i))

  This loss converts graded relevance to binary relevance by considering items
  with ``label >= 1`` as relevant and items with ``label < 1`` as non-relevant.

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional [..., list_size]-Array, indicating which items are valid
      for computing the loss. Items for which this is False will be ignored when
      computing the loss.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the weight for each item.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

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


def pointwise_mse_loss(scores: Array,
                       labels: Array,
                       *,
                       where: Optional[Array] = None,
                       weights: Optional[Array] = None,
                       reduce_fn: ReduceFn = jnp.sum) -> Array:
  r"""Mean squared error loss.

  Definition:

  .. math::
      \ell(s, y) = \sum_i (y_i - s_i)^2

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the loss. Items for which
      this is False will be ignored when computing the loss.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the weight for each item.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The mean squared error loss.
  """
  loss = jnp.square(scores - labels)

  if weights is not None:
    loss *= weights

  return utils.safe_reduce(loss, where=where, reduce_fn=reduce_fn)


def pairwise_mse_loss(scores: Array,
                      labels: Array,
                      *,
                      where: Optional[Array] = None,
                      weights: Optional[Array] = None,
                      reduce_fn: ReduceFn = jnp.sum) -> Array:
  r"""Pairwise mean squared error loss.

  Definition:

  .. math::
      \ell(s, y) =
      \sum_i \sum_j ((y_i - y_j) - (s_i - s_j))^2

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the loss. Items for which
      this is False will be ignored when computing the loss.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the weight for each item.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

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
