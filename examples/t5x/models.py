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

"""Ranking-specific encoder-decoder model and feature converter."""

import functools
import types
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import flax
import gin
import jax
import jax.numpy as jnp
import rax
import seqio
from t5x import metrics as metrics_lib
from t5x import models
from t5x import optimizers
import tensorflow as tf

FeatureSpec = seqio.FeatureConverter.FeatureSpec
PyTree = Any

# Set up default loss and metric functions. For this we will use the Rax softmax
# loss and several of the Rax metric functions. You can update these in the gin
# config file to different values.
DEFAULT_LOSS_FN = rax.softmax_loss

DEFAULT_METRIC_FNS = types.MappingProxyType({
    "ndcg": rax.ndcg_metric,
    "ndcg_at_10": functools.partial(rax.ndcg_metric, topn=10),
    "mrr": rax.mrr_metric,
    "mrr_at_10": functools.partial(rax.mrr_metric, topn=10),
})


@gin.configurable
class RankingEncDecFeatureConverter(seqio.FeatureConverter):
  """Feature converter for the `RankingEncDecModel`."""

  TASK_FEATURES = {
      "inputs": FeatureSpec(dtype=tf.int32, rank=2, sequence_dim=1),
      "targets": FeatureSpec(dtype=tf.int32, rank=2, sequence_dim=1),
      "label": FeatureSpec(dtype=tf.float32),
      "mask": FeatureSpec(dtype=tf.bool),
  }
  MODEL_FEATURES = {
      "encoder_input_tokens": FeatureSpec(
          dtype=tf.int32, rank=2, sequence_dim=1
      ),
      "decoder_input_tokens": FeatureSpec(
          dtype=tf.int32, rank=2, sequence_dim=1
      ),
      "decoder_target_tokens": FeatureSpec(
          dtype=tf.int32, rank=2, sequence_dim=1
      ),
      "label": FeatureSpec(dtype=tf.float32),
      "mask": FeatureSpec(dtype=tf.bool),
  }
  PACKING_FEATURE_DTYPES = None

  def _convert_features(
      self,
      ds: tf.data.Dataset,
      task_feature_lengths: Mapping[str, Union[int, Sequence[int]]],
  ) -> tf.data.Dataset:
    """Convert the dataset to be fed to the encoder-decoder model.

    This method adds a mask to indicate valid items in a list and additionally
    pads and truncates the task features to (list_size, sequence_length), so
    that it can be fed to the ranking encoder-decoder model.

    Args:
      ds: The dataset to convert.
      task_feature_lengths: A mapping indicating the sequence length of each
        feature.

    Returns:
      The converted dataset.
    """

    # Convert ragged tensors to dense tensors. The seqio `trim_and_pad_dataset`
    # cannot handle ragged tensors, so we convert them to dense here.
    def to_dense(features):
      return {
          key: (
              value.to_tensor(0)
              if isinstance(value, tf.RaggedTensor)
              else value
          )
          for key, value in features.items()
      }

    ds = ds.map(to_dense, num_parallel_calls=tf.data.AUTOTUNE)

    # Trim and pad the list_size dimension.
    ds = seqio.utils.trim_and_pad_dataset(
        ds,
        {
            "label": task_feature_lengths["label"][0],
            "mask": task_feature_lengths["label"][0],
            "targets": task_feature_lengths["targets"][0],
            "inputs": task_feature_lengths["inputs"][0],
        },
    )

    # Function to swap leading axes for "inputs" and "targets" so we can trim
    # and pad the sequence length. The seqio `trim_and_pad_dataset` can only pad
    # and truncate the leading dimension so this is needed to trim the sequence
    # length, which is the second dimension.
    def transpose_inputs_and_targets(task_features):
      return {
          **task_features,
          "inputs": tf.transpose(task_features["inputs"], [1, 0]),
          "targets": tf.transpose(task_features["targets"], [1, 0]),
      }

    # Trim and pad the sequence length dimension. This first swaps the sequence
    # length to the first dimension, then trims and pads it, then swaps the
    # sequence length back to the second dimension.
    ds = ds.map(transpose_inputs_and_targets)
    ds = seqio.utils.trim_and_pad_dataset(
        ds,
        {
            "targets": task_feature_lengths["targets"][1],
            "inputs": task_feature_lengths["inputs"][1],
        },
    )
    ds = ds.map(transpose_inputs_and_targets)

    # Finally, this adds the actual model features to the dataset and returns
    # the result. Note that the model is only predicting a single target, so
    # there is no need to construct autoregressive decoder inputs and we can
    # just use the default decoder input (0).
    def add_features(task_features):
      return {
          "encoder_input_tokens": task_features["inputs"],
          "decoder_input_tokens": tf.zeros_like(task_features["targets"]),
          "decoder_target_tokens": task_features["targets"],
          "label": task_features["label"],
          "mask": task_features["mask"],
      }

    ds = ds.map(add_features)
    return ds

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]
  ) -> Mapping[str, int]:
    """Define the length relationship between input and output features.

    Args:
      task_feature_lengths: A mapping indicating the task feature lengths.

    Returns:
      A mapping indicating the model feature lengths.
    """
    return {
        "encoder_input_tokens": task_feature_lengths["inputs"],
        "decoder_input_tokens": task_feature_lengths["targets"],
        "decoder_target_tokens": task_feature_lengths["targets"],
        "label": task_feature_lengths["label"],
        "mask": task_feature_lengths["label"],
    }


@gin.configurable
class RankingEncDecModel(models.EncoderDecoderModel):
  """EncoderDecoderModel for ranking data.

  As opposed to standard T5X EncoderDecoder model, this model supports batches
  with leading dimensions (batch_size, list_size, ...). This allows it to handle
  listwise ranking data, which makes it possible to compute ranking losses and
  metrics.
  """

  FEATURE_CONVERTER_CLS = RankingEncDecFeatureConverter

  def __init__(
      self,
      module: flax.linen.Module,
      input_vocabulary: seqio.Vocabulary,
      output_vocabulary: seqio.Vocabulary,
      optimizer_def: optimizers.OptimizerDefType,
      rax_loss_fn: rax.types.LossFn = DEFAULT_LOSS_FN,
      rax_metric_fns: Mapping[str, rax.types.MetricFn] = DEFAULT_METRIC_FNS,
      loss_normalizing_factor: Optional[float] = None,
  ):
    super().__init__(
        module,
        input_vocabulary,
        output_vocabulary,
        optimizer_def,
        loss_normalizing_factor=loss_normalizing_factor,
    )
    self._rax_loss_fn = rax_loss_fn
    self._rax_metric_fns = rax_metric_fns

  def get_initial_variables(
      self,
      rng: jax.random.PRNGKeyArray,
      input_shapes,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      input_types,
  ):
    """Initializes model variables for the given input shapes and types.

    This method supports computing variables for a batch of data of shape
    (batch_size, list_size, ...) by first flattening the leading batch
    dimensions.

    Args:
      rng: RNG Key used for initializing the model variables.
      input_shapes: Shapes for an input batch of data.
      input_types: Data types for an input batch of data.

    Returns:
      The initialized model variables.
    """
    # Flatten (batch_size, list_size, ...) into (batch_size * list_size, ...)
    # for model inputs.
    batch_size, list_size, *_ = input_shapes["encoder_input_tokens"]
    input_shapes = {
        **input_shapes,
        "encoder_input_tokens": (batch_size * list_size,) + input_shapes[
            "encoder_input_tokens"
        ][2:],
        "decoder_input_tokens": (batch_size * list_size,) + input_shapes[
            "decoder_input_tokens"
        ][2:],
    }
    return super().get_initial_variables(rng, input_shapes, input_types)

  def _compute_logits(
      self, params: PyTree, batch: Mapping[str, jnp.ndarray], *args, **kwargs
  ):
    """Computes logits on a batch of data.

    This method supports computing logits for a batch of data of shape
    (batch_size, list_size, ...) by first flattening the leading batch
    dimensions and then feeding it to the model.

    Args:
      params: The parameters of the model.
      batch: The batch of data to compute the logits for.
      *args: Positional arguments to pass to the parent `_compute_logits`.
      **kwargs: Keyword arguments to pass to the parent `_compute_logits`.

    Returns:
      The logits computed on the given batch of data.
    """
    # Flatten (batch_size, list_size, ...) into (batch_size * list_size, ...)
    # for model inputs.
    batch_size, list_size, *_ = batch["encoder_input_tokens"].shape
    flattened_batch = {
        **batch,
        "encoder_input_tokens": jnp.reshape(
            batch["encoder_input_tokens"],
            (batch_size * list_size,) + batch["encoder_input_tokens"].shape[2:],
        ),
        "decoder_input_tokens": jnp.reshape(
            batch["decoder_input_tokens"],
            (batch_size * list_size,) + batch["decoder_input_tokens"].shape[2:],
        ),
        "decoder_target_tokens": jnp.reshape(
            batch["decoder_target_tokens"],
            (batch_size * list_size,)
            + batch["decoder_target_tokens"].shape[2:],
        ),
    }

    # Compute logits on flattened inputs.
    output = super()._compute_logits(params, flattened_batch, *args, **kwargs)

    # Reshape output logits back to (batch_size, list_size, ...)
    output = jnp.reshape(output, (batch_size, list_size) + output.shape[1:])  # pytype: disable=attribute-error  # jax-ndarray

    # Compute per-item scores. We do three vmaps here for each of the dimensions
    # (batch_size, list_size, sequence_length, ...)
    output = jax.vmap(jax.vmap(jax.vmap(jnp.take)))(
        output, batch["decoder_target_tokens"]
    )
    output = jnp.squeeze(output, -1)
    return output

  def loss_fn(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray],
  ) -> Tuple[jnp.ndarray, metrics_lib.MetricsMap]:
    """Ranking loss function.

    Args:
      params: The parameters of the model.
      batch: The batch of data to compute the loss on.
      dropout_rng: An optional RNG key for dropout.

    Returns:
      A tuple (loss, metrics) containing the ranking loss and various ranking
      metrics.
    """
    scores = self._compute_logits(params, batch, dropout_rng)
    labels = batch["label"]
    mask = batch["mask"]

    # Compute ranking loss with Rax.
    loss = self._rax_loss_fn(scores, labels, where=mask, reduce_fn=jnp.sum)
    if self._loss_normalizing_factor is not None:
      loss = loss / self._loss_normalizing_factor

    # Compute ranking metrics.
    metrics = self._compute_metrics(loss, scores, labels, mask)

    return loss, metrics

  def _compute_metrics(
      self,
      loss: jnp.ndarray,
      scores: jnp.ndarray,
      labels: jnp.ndarray,
      mask: Optional[jnp.ndarray],
  ) -> metrics_lib.MetricsMap:
    """Computes ranking metrics.

    Args:
      loss: The loss for this batch.
      scores: The scores computed on this batch.
      labels: The relevance labels.
      mask: A mask indicating which items are valid.

    Returns:
      A dictionary containing various ranking metrics.
    """
    valid_items = jnp.mean(jnp.sum(jnp.int32(mask), axis=-1))
    labels_mean = jnp.mean(labels, where=mask)
    scores_mean = jnp.mean(scores, where=mask)
    metrics = {
        "loss": metrics_lib.AveragePerStep(total=loss),
        "debug/labels_mean": metrics_lib.AveragePerStep(total=labels_mean),
        "debug/scores_mean": metrics_lib.AveragePerStep(total=scores_mean),
        "debug/valid_items": metrics_lib.AveragePerStep(total=valid_items),
        "timing/steps_per_second": metrics_lib.StepsPerTime.from_model_output(),
        "timing/seconds": metrics_lib.Time(),
    }
    metrics.update(
        {
            f"metrics/{key}": metrics_lib.AveragePerStep(
                total=metric(scores, labels, where=mask)
            )
            for key, metric in self._rax_metric_fns.items()
        }
    )
    return metrics
