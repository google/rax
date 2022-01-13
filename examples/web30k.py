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

"""Example of training a neural ranking model on MSLR-WEB30K.

Usage with example output:

$ blaze run -c opt //third_party/py/rax/examples:web30k
epoch=1
  loss=4.46001
  metric/mrr=0.81766
  metric/ndcg=0.67343
  metric/ndcg@10=0.41583
epoch=2
  loss=4.44818
  metric/mrr=0.82619
  metric/ndcg=0.68358
  metric/ndcg@10=0.43313
epoch=3
  loss=4.44677
  metric/mrr=0.82789
  metric/ndcg=0.68388
  metric/ndcg@10=0.43411
"""

import collections
import functools
from typing import Mapping, Optional, Sequence, Tuple

from absl import app

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

import rax

# Used for loading data and data-preprocessing.
import tensorflow as tf
import tensorflow_datasets as tfds

# Type aliases
ModelState = flax.core.scope.FrozenVariableDict
OptState = optax.OptState


class DNN(nn.Module):
  """Implements a basic deep neural network for ranking."""

  @nn.compact
  def __call__(self, inputs):
    # Concatenate the features into a feature vector.
    x = [jnp.expand_dims(x, -1) for x in inputs.values()]
    x = jnp.concatenate(x, -1)

    # Perform log1p transformation on the features.
    x = jnp.sign(x) * jnp.log1p(jnp.abs(x))

    # Run inputs through several layers, finally producing a single score per
    # item.
    x = nn.Dense(64)(x)
    x = nn.relu(x)
    x = nn.Dense(32)(x)
    x = nn.relu(x)
    x = nn.Dense(1)(x)

    # Remove the feature axis since it is now a single score per item.
    x = jnp.squeeze(x, -1)
    return x


def prepare_dataset(ds: tf.data.Dataset,
                    batch_size: int = 128,
                    list_size: Optional[int] = 200,
                    shuffle_size: Optional[int] = 1000,
                    rng_seed: int = 42):
  """Prepares a training dataset by applying padding/truncating/etc."""
  tf.random.set_seed(rng_seed)
  ds = ds.cache()
  ds = ds.map(lambda e: {**e, "mask": tf.ones_like(e["label"], dtype=tf.bool)})
  if list_size is not None:
    pad = lambda t: tf.concat([t, tf.zeros(list_size, dtype=t.dtype)], -1)
    truncate = lambda t: t[:list_size]
    ds = ds.map(lambda e: tf.nest.map_structure(pad, e))
    ds = ds.map(lambda e: tf.nest.map_structure(truncate, e))
  if shuffle_size is not None:
    ds = ds.shuffle(shuffle_size, seed=rng_seed)
  ds = ds.padded_batch(batch_size)
  ds = ds.map(lambda e: (e, e.pop("label"), e.pop("mask")))
  ds = tfds.as_numpy(ds)
  return ds


def main(argv: Sequence[str]):
  del argv  # unused.

  # Load datasets.
  ds_train = prepare_dataset(tfds.load("mslr_web/30k_fold1", split="train"))

  # Create model and optimizer.
  model = DNN()
  optimizer = optax.adam(learning_rate=0.01)

  # Create Rax loss and metrics.
  loss_fn = jax.vmap(rax.softmax_loss)
  metric_fns = {
      "metric/mrr": jax.vmap(rax.mrr_metric),
      "metric/ndcg": jax.vmap(rax.ndcg_metric),
      "metric/ndcg@10": jax.vmap(functools.partial(rax.ndcg_metric, topn=10))
  }

  # Implement train and eval logic.
  @jax.jit
  def train_step(
      batch, model_state: ModelState,
      opt_state: OptState) -> Tuple[jnp.ndarray, ModelState, OptState]:
    # Unpack batch.
    inputs, labels, mask = batch

    # Compute gradients wrt model params
    def _loss_fn(params):
      scores = model.apply(model_state.copy({"params": params}), inputs)
      loss = jnp.mean(loss_fn(scores, labels, mask=mask))
      return loss

    params = model_state["params"]
    loss, grads = jax.value_and_grad(_loss_fn)(params)

    # Apply gradients using the optimizer.
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    model_state = model_state.copy({"params": params})
    return loss, model_state, opt_state

  @jax.jit
  def eval_step(batch, model_state: ModelState) -> Mapping[str, jnp.ndarray]:
    inputs, labels, mask = batch
    scores = model.apply(model_state, inputs)
    return {
        name: jnp.mean(metric_fn(scores, labels, mask=mask))
        for name, metric_fn in metric_fns.items()
    }

  # Initialize model and optimizer state.
  model_state = model.init(jax.random.PRNGKey(0), next(iter(ds_train))[0])
  opt_state = optimizer.init(model_state["params"])

  for epoch in range(3):
    metrics = collections.defaultdict(float)
    for batch in ds_train:
      # Perform train step and record loss.
      loss, model_state, opt_state = train_step(batch, model_state, opt_state)
      metrics["loss"] += loss

      # Perform eval and record metrics.
      for name, metric in eval_step(batch, model_state).items():
        metrics[name] += metric

    print(f"epoch={epoch + 1}")
    for name, metric in metrics.items():
      print(f"  {name}={metric / len(ds_train):.5f}")


if __name__ == "__main__":
  app.run(main)
