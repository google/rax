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

"""Function transformations for ranking losses and metrics.

These function transformations can be used to transform the ranking metrics and
losses. An example is ``approx_t12n`` which transforms a given ranking metric
into a ranking loss by plugging in differentiable approximations to the rank and
cutoff functions.

Example usage:

>>> scores = jnp.asarray([0., 1., 3., 2.])
>>> labels = jnp.asarray([0., 0., 1., 2.])
>>> approx_ndcg_loss_fn = rax.approx_t12n(rax.ndcg_metric)
>>> approx_ndcg_loss_fn(scores, labels)
DeviceArray(-0.71789175, dtype=float32)

"""

import functools
import inspect
from typing import Optional, TypeVar

import jax
import jax.numpy as jnp

from rax._src import utils
from rax._src.types import Array
from rax._src.types import LossFn
from rax._src.types import MetricFn

# Type aliases for ranking loss and metric functions.
LossOrMetricFn = TypeVar("LossOrMetricFn", LossFn, MetricFn)


def _accepts_args(fn, *args, **kwargs):
  """Returns True if `fn` can accept `*args` and `**kwargs`."""
  try:
    inspect.signature(fn).bind_partial(*args, **kwargs)
    return True
  except TypeError:
    return False


def approx_t12n(metric_fn: MetricFn, temperature: float = 1.0) -> LossFn:
  """Transforms ``metric_fn`` into an approximate differentiable loss.

  This transformation and uses a sigmoid approximation to compute ranks and
  indicators in metrics :cite:p:`qin2010general`. The returned approximate
  metric is mapped to negative values to be used as a loss.

  Example usage:

  >>> approx_mrr = rax.approx_t12n(rax.mrr_metric)
  >>> scores = jnp.asarray([0., 1., 3., 2.])
  >>> labels = jnp.asarray([0., 0., 1., 2.])
  >>> approx_mrr(scores, labels)
  DeviceArray(-0.6965873, dtype=float32)

  Example usage together with :func:`rax.gumbel_t12n`:

  >>> gumbel_approx_mrr = rax.gumbel_t12n(rax.approx_t12n(rax.mrr_metric))
  >>> scores = jnp.asarray([0., 1., 3., 2.])
  >>> labels = jnp.asarray([0., 0., 1., 2.])
  >>> gumbel_approx_mrr(scores, labels, key=jax.random.PRNGKey(42))
  DeviceArray(-0.71880937, dtype=float32)

  Args:
    metric_fn: The metric function to convert to an approximate loss.
    temperature: The temperature parameter to use for the sigmoid approximation.

  Returns:
    A loss function that computes the approximate version of `metric_fn`.
  """
  # Define step_fn for given temperature.
  step_fn = lambda x: jax.nn.sigmoid(x / temperature)

  # Construct kwargs for rank and cutoff functions.
  parameters = inspect.signature(metric_fn).parameters
  approx_kwargs = {}
  if "rank_fn" in parameters:
    approx_kwargs["rank_fn"] = functools.partial(
        utils.approx_ranks, step_fn=step_fn)
  if "cutoff_fn" in parameters:
    approx_kwargs["cutoff_fn"] = functools.partial(
        utils.approx_cutoff, step_fn=step_fn)

  @jax.util.wraps(metric_fn, namestr="approx_{fun}", docstr="Approx {doc}")
  def approx_metric_loss(scores, labels, **kwargs):
    # Use approx_kwargs by default but allow users to overwrite it (and add more
    # arguments) using the specified kwargs.
    kwargs = {**approx_kwargs, **kwargs}
    return -metric_fn(scores, labels, **kwargs)

  return approx_metric_loss


def bound_t12n(metric_fn: MetricFn):
  """Transforms ``metric_fn`` into a lower-bound differentiable loss.

  This transformation uses a hinge bound to compute ranks and indicators in
  metrics. The returned lower-bound of the metric is mapped to negative values
  to be used as a loss.

  Example usage:

  >>> bound_mrr = rax.bound_t12n(rax.mrr_metric)
  >>> scores = jnp.asarray([0., 1., 3., 2.])
  >>> labels = jnp.asarray([0., 1., 0., 1.])
  >>> bound_mrr(scores, labels)
  DeviceArray(-0.33333334, dtype=float32)

  Example usage together with :func:`rax.gumbel_t12n`:

  >>> gumbel_bound_mrr = rax.gumbel_t12n(rax.bound_t12n(rax.mrr_metric))
  >>> scores = jnp.asarray([0., 1., 3., 2.])
  >>> labels = jnp.asarray([0., 1., 0., 1.])
  >>> gumbel_bound_mrr(scores, labels, key=jax.random.PRNGKey(42))
  DeviceArray(-0.31619418, dtype=float32)

  Args:
    metric_fn: The metric function to convert to a lower-bound loss.

  Returns:
    A loss function that computes the lower-bound version of ``metric_fn``.
  """
  # Define lower and upper bound step_fn.
  upper_bound_step_fn = lambda x: jax.nn.relu(x + 1.)
  lower_bound_step_fn = lambda x: 1. - jax.nn.relu(1. - x)

  # Construct kwargs for rank and cutoff functions.
  parameters = inspect.signature(metric_fn).parameters
  approx_kwargs = {}
  if "rank_fn" in parameters:
    approx_kwargs["rank_fn"] = functools.partial(
        utils.approx_ranks, step_fn=upper_bound_step_fn)
  if "cutoff_fn" in parameters:
    approx_kwargs["cutoff_fn"] = functools.partial(
        utils.approx_cutoff, step_fn=lower_bound_step_fn)

  @jax.util.wraps(metric_fn, namestr="bounded_{fun}", docstr="Bounded {doc}")
  def bounded_metric_loss(scores, labels, **kwargs):
    # Use approx_kwargs by default but allow users to overwrite it (and add more
    # arguments) using the specified kwargs.
    kwargs = {**approx_kwargs, **kwargs}
    return -metric_fn(scores, labels, **kwargs)

  return bounded_metric_loss


def gumbel_t12n(loss_or_metric_fn: LossOrMetricFn,
                *,
                samples: int = 8,
                beta: float = 1.0,
                smoothing_factor: Optional[float] = None) -> LossOrMetricFn:
  """Transforms ``loss_or_metric_fn`` to operate on Gumbel-sampled scores.

  This transformation changes given ``loss_or_metric_fn`` so that it samples
  scores from a Gumbel distribution prior to computing the loss or metric
  :cite:p:`bruch2020stochastic`. The returned function requires a new ``key``
  keyword argument.

  Example usage:

  >>> loss_fn = rax.gumbel_t12n(rax.softmax_loss)
  >>> scores = jnp.asarray([0., 1., 3., 2.])
  >>> labels = jnp.asarray([0., 0., 1., 2.])
  >>> loss_fn(scores, labels, key=jax.random.PRNGKey(42))
  DeviceArray(6.2066536, dtype=float32)
  >>> loss_fn(scores, labels, key=jax.random.PRNGKey(79))
  DeviceArray(5.0127797, dtype=float32)

  Args:
    loss_or_metric_fn: A Rax loss or metric function.
    samples: Number of Gumbel samples to create.
    beta: Shape of the Gumbel distribution (default 1.0).
    smoothing_factor: If supplied, this will apply an extra
      ``log(softmax(scores) + smoothing_factor)`` transformation to the scores.
      If set to 1e-20, this effectively makes the loss compatible with the
      TF-Ranking versions of Gumbel losses. If ``smoothing_factor <= 0``, this
      may produce ``NaN`` values.

  Returns:
    A new function that behaves the same as ``loss_or_metric_fn`` but which
    requires an additional ``key`` argument that will be used to randomly sample
    the scores from a Gumbel distribution.
  """

  def expand_and_repeat_dim(a: Array, axis: int = 0):
    return jnp.repeat(jnp.expand_dims(a, axis), samples, axis)

  @jax.util.wraps(
      loss_or_metric_fn, namestr="gumbel_{fun}", docstr="Gumbel {doc}")
  def _loss_or_metric_fn_with_gumbel_scores(scores: Array, labels: Array, *,
                                            key: Array, **kwargs):
    # Repeat scores and labels `n` times by adding a new batch dim.
    scores = expand_and_repeat_dim(scores)
    labels = expand_and_repeat_dim(labels)

    # Also repeat other Array-type kwargs such as `where`, `weights`, etc.
    kwargs = {
        name: expand_and_repeat_dim(arg) if isinstance(arg, Array) else arg
        for name, arg in kwargs.items()
    }

    # Check if `loss_or_metric_fn` accepts a `key` argument. If it does, split
    # the key for downstream random ops.
    if _accepts_args(loss_or_metric_fn, key=key):
      key, kwargs["key"] = jax.random.split(key)

    # Update scores by drawing a sample from the gumbel distribution.
    gumbel_sample = jax.random.gumbel(key, shape=scores.shape)
    gumbel_scores = gumbel_sample * beta + scores

    if smoothing_factor is not None:
      # We intentionally do `log(softmax(x) + smoothing_factor)` instead of the
      # more numerically stable `jax.nn.log_softmax` as the former seems to have
      # a beneficial smoothing effect for ranking use-cases.
      gumbel_scores = jax.nn.softmax(
          gumbel_scores,
          where=kwargs.get("where", None),
          initial=jnp.min(scores))
      gumbel_scores = jnp.log(gumbel_scores + smoothing_factor)

    return loss_or_metric_fn(gumbel_scores, labels, **kwargs)

  return _loss_or_metric_fn_with_gumbel_scores
