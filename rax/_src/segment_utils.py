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

"""Utilities for segmented functionality."""

from typing import Optional, Union

import jax
import jax.numpy as jnp

from rax._src.types import Array


def same_segment_mask(segments: Array) -> Array:
  """Returns an array indicating whether a pair is in the same segment."""
  return jnp.expand_dims(segments, -1) == jnp.expand_dims(segments, axis=-2)


def segment_sum(
    a: Array, segments: Array, where: Optional[Array] = None
) -> Array:
  """Returns segment sum."""
  if where is not None:
    where = jnp.expand_dims(where, -1) & jnp.expand_dims(where, -2)
  return jnp.sum(
      jnp.expand_dims(a, -2) * jnp.int32(same_segment_mask(segments)),
      axis=-1,
      where=where,
  )


def segment_max(
    a: Array,
    segments: Array,
    where: Optional[Array] = None,
    initial: Optional[Union[float, int]] = None,
) -> Array:
  """Returns segment max."""
  mask = same_segment_mask(segments)
  if where is not None:
    mask &= jnp.expand_dims(where, -1) & jnp.expand_dims(where, -2)
  initial = jnp.min(a) if initial is None else initial
  return jnp.max(
      jnp.broadcast_to(jnp.expand_dims(a, -2), mask.shape),
      axis=-1,
      where=mask,
      initial=initial
  )


def segment_log_softmax(
    a: Array,
    segments: Array,
    where: Optional[Array] = None
) -> Array:
  """Returns segment log-softmax."""
  a_max = segment_max(a, segments, where=where, initial=jnp.min(a))
  shifted = a - jax.lax.stop_gradient(a_max)
  shifted_logsumexp = jnp.log(
      segment_sum(jnp.exp(shifted), segments, where=where)
  )
  return shifted - shifted_logsumexp


def segment_softmax(
    a: Array,
    segments: Array,
    where: Optional[Array] = None
) -> Array:
  """Returns segment softmax."""
  a_max = segment_max(a, segments, where=where, initial=jnp.min(a))
  unnormalized = jnp.exp(a - jax.lax.stop_gradient(a_max))
  return unnormalized / segment_sum(unnormalized, segments, where=where)


def in_segment_indices(segments: Array) -> Array:
  """Returns 0-based indices per segment.

  For example: segments = [0, 0, 0, 1, 2, 2], then the in-segment indices are
  [0, 1, 2 | 0 | 0, 1], where we use "|" to mark the boundaries of the segments.
  Returns [0, 1, 2, 0, 0, 1] for segments [0, 0, 0, 1, 2, 2].

  Args:
    segments: A :class:`jax.numpy.ndarray` to indicate segments of items that
      should be grouped together. Like ``[0, 0, 1, 0, 2]``. The segments may or
      may not be sorted.

  Returns:
    An Array with 0-based indices per segment.
  """
  same_segments = jnp.int32(same_segment_mask(segments))
  lower_triangle = jnp.tril(jnp.ones_like(same_segments))
  return jnp.sum(same_segments * lower_triangle, axis=-1) - 1


def first_item_segment_mask(
    segments: Array, where: Optional[Array] = None
) -> Array:
  """Constructs a mask that selects the first item per segment.

  Args:
    segments: A :class:`jax.numpy.ndarray` to indicate segments of items that
      should be grouped together. Like ``[0, 0, 1, 0, 2]``. The segments may or
      may not be sorted.
    where: An optional :class:`jax.numpy.ndarray` to indicate invalid items.

  Returns:
    A :class:`jax.numpy.ndarray` of the same shape as ``segments`` that selects
    the first valid item in each segment.
  """
  # Construct a same-segment mask.
  mask = same_segment_mask(segments)

  # Mask out invalid items.
  if where is not None:
    mask = mask & (jnp.expand_dims(where, -1) & jnp.expand_dims(where, -2))

  # Remove duplicated columns in the mask so only the first item for each
  # segment appears in the result.
  mask = mask & (jnp.cumsum(mask, axis=-1) == 1)

  # Collapse mask to original `segments` shape, so we get a mask that selects
  # exactly the first item per segment.
  return jnp.any(mask, axis=-2)

