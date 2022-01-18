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

"""Example of training a linear model on MSLR-WEB10K with approximate metrics.

Usage with example output:

$ blaze run -c opt //third_party/py/rax/examples:approx_metrics
             | AP      | NDCG    | R@50
ApproxAP     | 0.59480 | 0.65895 | 0.58062
ApproxNDCG   | 0.58769 | 0.67001 | 0.57444
ApproxR@50   | 0.58794 | 0.64281 | 0.58129
"""

import functools
from typing import Optional, Sequence

from absl import app
from clu.metrics import Average
import jax
import jax.numpy as jnp
import rax

# Used for loading data and data-preprocessing.
import tensorflow as tf
import tensorflow_datasets as tfds


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


@jax.jit
def model_fn(w, features):
  """Computes the model scores for the given weights `w`.

  Args:
    w: The weights of the model.
    features: The features as input to the model.

  Returns:
    The model scores.
  """
  log1p = lambda x: jnp.sign(x) * jnp.log1p(jnp.abs(x))
  features = jnp.concatenate(
      [jnp.expand_dims(f, axis=-1) for f in features.values()], axis=-1)
  return jnp.dot(log1p(features), w)


def train(ds, approx_metric, epochs: int = 10, lr: float = 10.0, seed: int = 7):
  """Trains a model using the given `approx_metric`.

  Args:
    ds: The dataset to train on.
    approx_metric: The approx metric function to maximize.
    epochs: The number of epochs to run.
    lr: The learning rate to use.
    seed: Random seed to initialize the model weights.

  Returns:
    The learned weights.
  """
  # Initialize the model weights.
  number_of_features = len(next(iter(ds))[0])
  w = jax.random.uniform(jax.random.PRNGKey(seed), (number_of_features,))

  # Define loss and gradient function.
  def loss_fn(w, batch):
    features, labels, mask = batch
    scores = model_fn(w, features)
    return approx_metric(scores, labels, mask=mask)

  grad_fn = jax.jit(jax.grad(loss_fn))

  # Optimize model with SGD and return the learned weights.
  for _ in range(epochs):
    for batch in ds:
      grads = grad_fn(w, batch)
      w = w - lr * grads
  return w


def eval_metrics(ds, w, metrics):
  """Evaluates a model on several metrics.

  Args:
    ds: The dataset to evaluate on.
    w: The model weights.
    metrics: A dict of metric callables.

  Returns:
    A dict with the same keys as metrics but with the computed metric values.
  """
  # Define initial metric values.
  metric_values = {
      metric_name: Average(jnp.float32(0.), jnp.int32(0))
      for metric_name in metrics
  }

  @jax.jit
  def update_metric_values(batch, metric_values):
    features, labels, mask = batch
    scores = model_fn(w, features)
    for metric_name, metric in metrics.items():
      metric_values[metric_name] = metric_values[metric_name].merge(
          Average(metric(scores, labels, mask=mask), jnp.int32(1)))
    return metric_values

  # Iterate over each batch and update the metric average values.
  for batch in ds:
    metric_values = update_metric_values(batch, metric_values)

  # Return final metric values.
  return {
      metric_name: metric.compute()
      for metric_name, metric in metric_values.items()
  }


def main(argv: Sequence[str], epochs: int = 10):
  del argv  # Unused.

  # Load datasets.
  ds_train = prepare_dataset(tfds.load("mslr_web/10k_fold1", split="train"))

  # Build approx metrics. This uses the approx transformation to get an
  # approximate and differentiable version of the ranking metric. You can use
  # any of the Rax ranking metrics here. For this example, we will optimize AP,
  # NDCG and Recall@50.
  approx_ap = rax.approx_t12n(rax.ap_metric)
  approx_ndcg = rax.approx_t12n(rax.ndcg_metric)
  approx_recall_at_50 = rax.approx_t12n(
      functools.partial(rax.recall_metric, topn=50))

  # Train model with each of the approx metrics.
  w_ndcg = train(ds_train, approx_ndcg, epochs=epochs)
  w_ap = train(ds_train, approx_ap, epochs=epochs)
  w_recall = train(ds_train, approx_recall_at_50, epochs=epochs)

  # Evaluate each model on exact metrics and print a table of results:
  metrics = {
      "AP": rax.ap_metric,
      "NDCG": rax.ndcg_metric,
      "R@50": functools.partial(rax.recall_metric, topn=50)
  }
  metric_names = list(metrics.keys())
  print(f"{'':<12} | " + " | ".join(f"{name:<7}" for name in metric_names))

  def print_results(name, w):
    results = eval_metrics(ds_train, w, metrics)
    results = " | ".join(
        [f"{results[metric]:.5f}" for metric in metric_names])
    print(f"{name:<12} | {results}")

  print_results("ApproxAP", w_ap)
  print_results("ApproxNDCG", w_ndcg)
  print_results("ApproxR@50", w_recall)


if __name__ == "__main__":
  app.run(main)
