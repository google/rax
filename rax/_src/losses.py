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
>>> print(rax.softmax_loss(scores, labels))
1.4076059

Usage with a batch of data and a mask to indicate valid items.

>>> scores = jnp.array([[2., 1., 0.], [1., 0.5, 1.5]])
>>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
>>> where = jnp.array([[True, True, False], [True, True, True]])
>>> print(rax.pairwise_hinge_loss(
...     scores, labels, where=where, reduce_fn=jnp.mean))
0.16666667

To compute gradients of each loss function, please use standard JAX
transformations such as :func:`jax.grad` or :func:`jax.value_and_grad`:

>>> scores = jnp.asarray([[0., 1., 3.], [1., 2., 0.]])
>>> labels = jnp.asarray([[0., 0., 1.], [1., 0., 0.]])
>>> print(jax.grad(rax.softmax_loss)(scores, labels, reduce_fn=jnp.mean))
[[ 0.02100503  0.0570976  -0.07810265]
 [-0.37763578  0.33262047  0.04501529]]
"""

import operator
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from rax._src import metrics
from rax._src import utils
from rax._src.types import Array
from rax._src.types import LambdaweightFn
from rax._src.types import ReduceFn


def softmax_loss(scores: Array,
                 labels: Array,
                 *,
                 where: Optional[Array] = None,
                 weights: Optional[Array] = None,
                 label_fn: Callable[..., Array] = lambda a, where: a,
                 reduce_fn: Optional[ReduceFn] = jnp.mean) -> Array:
  r"""Softmax loss.

  Definition:

  .. math::
      \ell(s, y) =
      - \sum_i y_i \log \frac{\exp(s_i)}{\sum_j \exp(s_j)}

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
  # Applies mask so that masked elements do not count towards the loss.
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


def poly1_softmax_loss(scores: Array,
                       labels: Array,
                       *,
                       epsilon: float = 1.0,
                       where: Optional[Array] = None,
                       weights: Optional[Array] = None,
                       reduce_fn: Optional[ReduceFn] = jnp.mean) -> Array:
  r"""Poly1 softmax loss.

  Definition :cite:p:`leng2022polyloss`:

  .. math::
      \ell(s, y) = softmax(s, y) + \epsilon * (1 - pt)

  where :math:`softmax` is the standard softmax loss as implemented in
  :func:`~rax.softmax_loss` and :math:`pt` is the target softmax probability
  defined as:

  .. math::
      pt = \sum_i \frac{y_i}{\sum_j y_j} \frac{\exp(s_i)}{\sum_j \exp(s_j)}

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    epsilon: A float hyperparameter indicating the weight of the leading
      polynomial coefficient in the poly loss.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the loss. Items for which
      this is False will be ignored when computing the loss.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the weight for each item.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The poly1 softmax loss.
  """
  # Compute softmax cross-entropy loss without batch reduction.
  ce = softmax_loss(
      scores, labels, where=where, weights=weights, reduce_fn=None)

  # Applies mask so that masked elements do not count towards the loss.
  if where is not None:
    labels = jnp.where(where, labels, jnp.zeros_like(labels))
    scores = jnp.where(where, scores, -jnp.ones_like(scores) * jnp.inf)

  # Apply weights to labels.
  if weights is not None:
    labels *= weights

  # Compute target probabilities.
  scores_softmax = jax.nn.softmax(scores)
  labels_normalized = utils.normalize_probabilities(labels, where=where)
  pt = jnp.sum(labels_normalized * scores_softmax, where=where, axis=-1)

  # For lists where all items are masked, this sets pt to 1 so that the term
  # (1 - pt) is set to 0 for the loss computation.
  if where is not None:
    pt = jnp.where(jnp.all(jnp.logical_not(where), axis=-1), 1., pt)

  # Compute and return the poly1 loss.
  loss = ce + epsilon * (1. - pt)
  return utils.safe_reduce(loss, reduce_fn=reduce_fn)


def unique_softmax_loss(scores: Array,
                        labels: Array,
                        *,
                        where: Optional[Array] = None,
                        weights: Optional[Array] = None,
                        gain_fn: Optional[Callable[
                            [Array], Array]] = metrics.default_gain_fn,
                        reduce_fn: ReduceFn = jnp.mean) -> Array:
  r"""Unique softmax loss.

  Definition :cite:p:`zhu2020listwise`:

  .. math::
      \ell(s, y) =
      - \sum_i \operatorname{gain}(y_i)
      \log \frac{\exp(s_i)}{\exp(s_i) + \sum_{j : y_j < y_i} \exp(s_j)}

  where :math:`\operatorname{gain}(y_i)` is a user-specified gain function
  applied to label :math:`y_i` to boost items with higher relevance.

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
    gain_fn: An optional function that maps relevance labels to gain values. If
      provided, the per-item losses are multiplied by ``gain_fn(label)`` to
      boost the importance of relevant items.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The unique softmax loss.
  """
  # Construct pairwise matrices for scores and labels. The labels matrix will
  # indicate, for each item, which other items have a smaller label.
  labels_lt = jnp.expand_dims(labels, -2) < jnp.expand_dims(labels, -1)
  scores_repeated = jnp.repeat(
      jnp.expand_dims(scores, -2), scores.shape[-1], axis=-2)

  # Build an identity mask to select items during softmax computation.
  identity_mask = jnp.identity(scores.shape[-1], dtype=jnp.bool_)

  # Apply mask to ignore invalid items.
  if where is not None:
    labels_lt &= jnp.expand_dims(where, -2)
    identity_mask &= jnp.expand_dims(where, -2)

  # Compute log-softmax for each of the items' unique softmax distribution. This
  # effectively computes the log_softmax for each item using only itself and
  # items with a smaller relevance label in the denominator. The computed
  # log_softmax distribution is reduced by selecting its diagonal elements.
  log_softmax = jax.nn.log_softmax(
      scores_repeated,
      axis=-1,
      where=(identity_mask | labels_lt),
      initial=jnp.min(scores))
  log_softmax = jnp.diagonal(log_softmax, axis1=-2, axis2=-1)

  # Apply per-item weights.
  if weights is not None:
    log_softmax *= weights

  # Apply per-item `gain_fn` boost.
  if gain_fn is not None:
    log_softmax *= gain_fn(labels)

  # Compute per-list loss and return a reduced loss.
  loss = -jnp.sum(log_softmax, axis=-1, where=where)
  return utils.safe_reduce(loss, reduce_fn=reduce_fn)


def listmle_loss(scores: Array,
                 labels: Array,
                 *,
                 key: Optional[Array] = None,
                 where: Optional[Array] = None,
                 reduce_fn: Optional[ReduceFn] = jnp.mean) -> Array:
  r"""ListMLE Loss.

  .. note::

    This loss performs sorting using the given labels. If the labels contain
    multiple identical values, you should provide a :func:`~jax.random.PRNGKey`
    to the ``key`` argument to make sure ties are broken randomly during the
    sorting operation.

  Definition :cite:p:`xia2008listwise`:

  .. math::
      \ell(s, y) =
      - \sum_i \log
        \frac{\exp(s_i)}{\sum_j I[rank(y_j) \ge rank(y_i)] \exp(s_j)}

  where :math:`\operatorname{rank}(y_i)` indicates the rank of item :math:`i`
  after sorting all labels :math:`y`.

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    key: An optional :func:`~jax.random.PRNGKey` to perform random tie-breaking.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the loss. Items for which
      this is False will be ignored when computing the loss.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The listmle loss.
  """
  # Sort scores and mask by labels.
  if where is None:
    where = jnp.ones_like(scores, dtype=jnp.bool_)

  scores_sorted, where_sorted = utils.sort_by(
      labels, [scores, where], where=where, key=key)

  # Compute cumulative logsumexp.
  lse = utils.logcumsumexp(
      scores_sorted, axis=-1, where=where_sorted, reverse=True)

  # Reduce list MLE loss.
  loss = -jnp.sum(scores_sorted - lse, axis=-1, where=where_sorted)
  return utils.safe_reduce(loss, reduce_fn=reduce_fn)


def pairwise_loss(scores: Array,
                  labels: Array,
                  *,
                  pair_loss_fn: Callable[[Array, Array], Tuple[Array, Array]],
                  lambdaweight_fn: Optional[LambdaweightFn] = None,
                  where: Optional[Array] = None,
                  weights: Optional[Array] = None,
                  reduce_fn: Optional[ReduceFn] = jnp.mean) -> Array:
  r"""Generic pairwise loss.

  The ``pair_loss_fn`` takes ``(scores_diff, labels_diff)`` and returns the loss
  for each pair and also the valid pairs considered in the loss.

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    pair_loss_fn: A function that outputs ``(pair_losses, valid_pairs)`` given
      ``(scores_diff, labels_diff)``.
    lambdaweight_fn: An optional function that outputs lambdaweights.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the loss. Items for which
      this is False will be ignored when computing the loss.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the weight for each item.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The pairwise loss.
  """
  # Expand scores and labels into pairwise versions.
  scores_diff = utils.compute_pairs(scores, operator.sub)
  labels_diff = utils.compute_pairs(labels, operator.sub)

  # Compute losses and validity of all pairs.
  pair_losses, valid_pairs = pair_loss_fn(scores_diff, labels_diff)

  # Apply mask to valid pairs.
  if where is not None:
    valid_pairs &= utils.compute_pairs(where, operator.and_)

  # Apply weights to losses.
  if weights is not None:
    pair_losses *= utils.compute_pairs(weights, lambda x, y: x)

  # Apply lambda weights to losses.
  if lambdaweight_fn is not None:
    pair_losses *= lambdaweight_fn(
        scores, labels, where=where, weights=weights)

  return utils.safe_reduce(pair_losses, where=valid_pairs, reduce_fn=reduce_fn)


def pairwise_hinge_loss(scores: Array,
                        labels: Array,
                        *,
                        where: Optional[Array] = None,
                        weights: Optional[Array] = None,
                        lambdaweight_fn: Optional[LambdaweightFn] = None,
                        reduce_fn: Optional[ReduceFn] = jnp.mean) -> Array:
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
    lambdaweight_fn: An optional function that outputs lambdaweights.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The pairwise hinge loss.
  """

  def _hinge_loss(scores_diff: Array,
                  labels_diff: Array) -> Tuple[Array, Array]:
    return jax.nn.relu(1. - scores_diff), labels_diff > 0

  return pairwise_loss(
      scores,
      labels,
      pair_loss_fn=_hinge_loss,
      where=where,
      weights=weights,
      lambdaweight_fn=lambdaweight_fn,
      reduce_fn=reduce_fn)


def pairwise_logistic_loss(scores: Array,
                           labels: Array,
                           *,
                           where: Optional[Array] = None,
                           weights: Optional[Array] = None,
                           lambdaweight_fn: Optional[LambdaweightFn] = None,
                           reduce_fn: Optional[ReduceFn] = jnp.mean) -> Array:
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
    lambdaweight_fn: An optional function that outputs lambdaweights.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The pairwise logistic loss.
  """

  def _logistic_loss(scores_diff: Array,
                     labels_diff: Array) -> Tuple[Array, Array]:
    return (jax.nn.relu(-scores_diff) +
            jnp.log1p(jnp.exp(-jnp.abs(scores_diff))), labels_diff > 0)

  return pairwise_loss(
      scores,
      labels,
      pair_loss_fn=_logistic_loss,
      where=where,
      weights=weights,
      lambdaweight_fn=lambdaweight_fn,
      reduce_fn=reduce_fn)


def pointwise_sigmoid_loss(scores: Array,
                           labels: Array,
                           *,
                           where: Optional[Array] = None,
                           weights: Optional[Array] = None,
                           reduce_fn: Optional[ReduceFn] = jnp.mean) -> Array:
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
                       reduce_fn: Optional[ReduceFn] = jnp.mean) -> Array:
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
                      lambdaweight_fn: Optional[LambdaweightFn] = None,
                      reduce_fn: Optional[ReduceFn] = jnp.mean) -> Array:
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
    lambdaweight_fn: An optional function that outputs lambdaweights.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The pairwise mean squared error loss.
  """

  def _mse_loss(scores_diff: Array, labels_diff: Array) -> Tuple[Array, Array]:
    return (jnp.square(scores_diff - labels_diff),
            jnp.ones_like(labels_diff > 0))

  return pairwise_loss(
      scores,
      labels,
      pair_loss_fn=_mse_loss,
      where=where,
      weights=weights,
      lambdaweight_fn=lambdaweight_fn,
      reduce_fn=reduce_fn)
