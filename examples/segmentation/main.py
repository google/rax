# Copyright 2025 Google LLC.
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

"""Example of using segmentation for efficient packing of lists."""

import functools
import json
from typing import Sequence

from absl import app
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax
import rax
import tensorflow as tf
import tensorflow_datasets as tfds


def read_data(batch_size: int = 1024, lists_per_batch: int = 64):
  """Reads ranking data in a segmented format.

  This produces batches of flat data with a segment array to indicate which
  items should be grouped together to form lists:

  ```
  {
    "float_features": jax.Array(shape=[n, n_features]),
    "labels": jax.Array(shape=[n]),
    "segments": jax.Array(shape=[n]),
    "mask": jax.Array(shape=[n]),
  }
  ```

  where ``n`` is the batch size.

  Args:
    batch_size: The batch size (in number of items, not lists).
    lists_per_batch: The number of lists to pack into a single batch. Should be
      roughly `batch_size / max_num_items_per_list` for efficiency.

  Returns:
    A `tf.data.Dataset` with packed ranking examples.
  """

  # Helper function to construct segmented data from RaggedTensor data.
  def to_segments(e, max_size: int = 1024):
    # Get features and truncate them to max_size to get segmented data versions.
    float_features = e["float_features"].values[:max_size]
    segments = e["label"].value_rowids()[:max_size]
    labels = e["label"].values[:max_size]
    # Return result but remove the last segment as it may have been cutoff with
    # the above truncation operation.
    return {
        "float_features": float_features,
        "labels": labels,
        "segments": tf.where(
            segments == segments[-1], -tf.ones_like(segments), segments
        ),
        "mask": tf.where(
            segments == segments[-1],
            tf.zeros_like(segments, dtype=tf.bool),
            tf.ones_like(segments, dtype=tf.bool),
        ),
    }

  # Read data from TFDS.
  tf.random.set_seed(42)
  ds = tfds.load("mslr_web/30k_fold1", split="train")
  ds = ds.cache()
  ds = ds.repeat()

  # Convert data to RaggedTensors using `lists_per_batch`. This determines how
  # many lists will be packed into a batch.
  ds = ds.ragged_batch(lists_per_batch)

  # Map to segmented representation. This also truncates the segmented
  # representation if there are too many items per batch.
  ds = ds.map(lambda x: to_segments(x, max_size=batch_size))

  # Pad to batch_size if there were not enough items per batch.
  ds = ds.padded_batch(1, padded_shapes={
      "float_features": (batch_size, -1),
      "labels": (batch_size,),
      "segments": (batch_size,),
      "mask": (batch_size,),
  }).unbatch()
  return ds


class DNN(nn.Module):
  """Implements a basic deep neural network for ranking."""

  @nn.compact
  def __call__(self, inputs):
    x = inputs["float_features"]

    # Perform log1p transformation on the features.
    x = jnp.sign(x) * jnp.log1p(jnp.abs(x))

    # Run inputs through.
    x_hidden = nn.Dense(64)(x)
    x_hidden = nn.relu(x_hidden)
    x = nn.Dense(1)(jnp.concatenate([x, x_hidden], axis=-1))

    # Remove the feature axis since it is now a single score per item.
    x = jnp.squeeze(x, -1)
    return x


def main(argv: Sequence[str], steps: int = 600, steps_per_eval: int = 200):
  del argv  # Unused.

  # Read dataset.
  ds = tfds.as_numpy(read_data(batch_size=1024))

  # Create model and optimizer.
  model = DNN()
  optimizer = optax.adam(learning_rate=0.001)

  # Initialize model and optimizer state.
  w = model.init(jax.random.PRNGKey(0), next(iter(ds)))
  opt_state = optimizer.init(w)

  # Create training step function.
  @jax.jit
  def train_step(w, opt_state, batch):
    def loss_fn(w):
      return rax.softmax_loss(
          scores=model.apply(w, batch),
          labels=batch["labels"],
          segments=batch["segments"],
          where=batch["mask"],
      )

    grads = jax.grad(loss_fn)(w)
    updates, opt_state = optimizer.update(grads, opt_state, w)
    w = optax.apply_updates(w, updates)
    return w, opt_state

  # Create eval step function.
  metric_fns = {
      "ndcg": rax.ndcg_metric,
      "loss": rax.softmax_loss,
      "ndcg@10": functools.partial(rax.ndcg_metric, topn=10),
      "mrr": rax.mrr_metric
  }
  @jax.jit
  def eval_step(w, batch):
    scores = model.apply(w, batch)
    labels, segments, mask = batch["labels"], batch["segments"], batch["mask"]
    return {
        name: metric_fn(scores, labels, segments=segments, where=mask)
        for name, metric_fn in metric_fns.items()
    }

  # Iterate dataset and optimize neural network while recording metrics.
  output = []
  metrics = {"loss": 0.0, "mrr": 0.0, "ndcg": 0.0, "ndcg@10": 0.0}
  for step, batch in zip(range(steps), ds):
    w, opt_state = train_step(w, opt_state, batch)
    metrics = {
        key: metrics[key] + float(value)
        for key, value in eval_step(w, batch).items()
    }
    if (step + 1) % steps_per_eval == 0:
      output.append({
          "step": step + 1,
          "loss": metrics["loss"] / steps_per_eval,
          "mrr": metrics["mrr"] / steps_per_eval,
          "ndcg": metrics["ndcg"] / steps_per_eval,
          "ndcg@10": metrics["ndcg@10"] / steps_per_eval,
      })
      metrics = {"loss": 0.0, "mrr": 0.0, "ndcg": 0.0, "ndcg@10": 0.0}

  # Print output as JSON.
  print(json.dumps(output, indent=2))


if __name__ == "__main__":
  app.run(main)
