# Copyright 2024 Google LLC.
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

# pytype: skip-file
"""Tests for rax.examples.t5x.models."""

from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from examples.t5x import models
import tensorflow as tf


class RankingEncDecFeatureConverterTest(absltest.TestCase):

  def test_end_to_end(self):
    # Create sample dataset.
    ds = {
        "inputs": tf.ragged.constant(
            [
                [5, 1024, 250, 1, 24, 205, 555],
                [5, 304, 200, 200],
                [1, 204, 333, 1024, 5],
            ],
            dtype=tf.int32,
        ),
        "targets": tf.ragged.constant([[10], [10], [10]], dtype=tf.int32),
        "label": tf.constant([0.0, 1.0, 0.0], dtype=tf.float32),
        "mask": tf.constant([True, True, True], dtype=tf.bool),
    }
    ds = tf.data.Dataset.from_tensors(ds)

    # Create feature converter.
    converter = models.RankingEncDecFeatureConverter(
        pack=False, apply_length_check=False
    )

    # Convert the sample dataset using the feature converter.
    task_feature_lengths = {
        "inputs": (4, 5),
        "targets": (4, 1),
        "label": (4,),
        "mask": (4,),
    }
    ds_converted = converter(ds, task_feature_lengths)

    # Get an output sample and validate that it is correct.
    sample = next(iter(ds_converted))
    expected = {
        "encoder_input_tokens": tf.constant(
            [
                [5, 1024, 250, 1, 24],
                [5, 304, 200, 200, 0],
                [1, 204, 333, 1024, 5],
                [0, 0, 0, 0, 0],
            ],
            dtype=tf.int32,
        ),
        "decoder_input_tokens": tf.constant(
            [[0], [0], [0], [0]], dtype=tf.int32
        ),
        "decoder_target_tokens": tf.constant(
            [[10], [10], [10], [0]], dtype=tf.int32
        ),
        "mask": tf.constant([True, True, True, False], dtype=tf.bool),
        "label": tf.constant([0.0, 1.0, 0.0, 0.0], dtype=tf.float32),
    }
    self.assertCountEqual(sample, expected)
    for key in expected:
      np.testing.assert_equal(sample[key].numpy(), expected[key].numpy())

    # Validate that `get_model_feature_lengths` is correct as well.
    model_feature_lengths = converter.get_model_feature_lengths(
        task_feature_lengths
    )
    for key, expected_shape in model_feature_lengths.items():
      self.assertEqual(sample[key].shape, expected_shape)


class RankingEncDecModelTest(absltest.TestCase):

  def test_get_initial_variables(self):
    # Create a RankingEncDecModel with mocked implementations.
    mocked_module = mock.Mock(spec=models.flax.linen.Module)
    mocked_module.init = mock.Mock(return_value={})
    mocked_vocab = mock.Mock(spec=models.seqio.Vocabulary)
    mocked_optimizer = mock.Mock(spec=models.optimizers.OptimizerDefType)
    model = models.RankingEncDecModel(
        mocked_module, mocked_vocab, mocked_vocab, mocked_optimizer
    )

    # Call method to initialize variables.
    rng = jax.random.PRNGKey(0)
    input_shapes = {
        "encoder_input_tokens": (16, 4, 5),
        "decoder_input_tokens": (16, 4, 1),
        "decoder_target_tokens": (16, 4, 1),
        "mask": (
            16,
            4,
        ),
        "label": (
            16,
            4,
        ),
    }
    input_types = {
        "encoder_input_tokens": jax.numpy.int32,
        "decoder_input_tokens": jax.numpy.int32,
        "decoder_target_tokens": jax.numpy.int32,
        "mask": jax.numpy.bool_,
        "label": jax.numpy.float32,
    }
    model.get_initial_variables(rng, input_shapes, input_types)

    # Validate that the Flax module.init function is called with appropriate
    # initialization args where the leading (batch_size, list_size, ...)
    # dimensions are flattened to (batch_size * list_size, ...).
    mocked_module.init.assert_called()
    args = mocked_module.init.call_args.args
    np.testing.assert_equal(np.array(args[0]), np.array(rng))
    self.assertEqual(args[1].shape, (16 * 4, 5))  # encoder_input_tokens
    self.assertEqual(args[2].shape, (16 * 4, 1))  # decoder_input_tokens
    self.assertEqual(args[3].shape, (16 * 4, 1))  # decoder_target_tokens

  def test_loss_fn(self):
    # Create a RankingEncDecModel with mocked implementations.
    mocked_module = mock.Mock(spec=models.flax.linen.Module)
    mocked_vocab = mock.Mock(spec=models.seqio.Vocabulary)
    mocked_optimizer = mock.Mock(spec=models.optimizers.OptimizerDefType)
    model = models.RankingEncDecModel(
        mocked_module, mocked_vocab, mocked_vocab, mocked_optimizer
    )

    batch_key, return_key = jax.random.split(jax.random.PRNGKey(0))
    params = {}
    batch = {
        "encoder_input_tokens": jnp.ones((16, 4, 5), dtype=jnp.int32),
        "decoder_input_tokens": jnp.ones((16, 4, 1), dtype=jnp.int32),
        "decoder_target_tokens": jnp.ones((16, 4, 1), dtype=jnp.int32),
        "mask": jnp.ones(
            (
                16,
                4,
            ),
            dtype=jnp.bool_,
        ),
        "label": jnp.float32(
            jax.random.bernoulli(
                batch_key,
                0.2,
                (
                    16,
                    4,
                ),
            )
        ),
    }

    # Set a random return value of the correct shape for the module apply
    # function.
    mocked_module.apply.return_value = jax.random.uniform(
        return_key, (16 * 4, 1, 20)
    )

    # Call the loss_fn on the model.
    loss, metrics = model.loss_fn(params, batch, None)

    # Check that the Flax module.apply function is called with appropriate call
    # args where the leading (batch_size, list_size, ...) dimensions are
    # flattened to (batch_size * list_size, ...).
    args = mocked_module.apply.call_args.args
    self.assertEqual(args[0], {"params": params})
    self.assertEqual(args[1].shape, (16 * 4, 5))  # encoder_input_tokens
    self.assertEqual(args[2].shape, (16 * 4, 1))  # decoder_input_tokens
    self.assertEqual(args[3].shape, (16 * 4, 1))  # decoder_target_tokens

    # Check the loss and metric values.
    np.testing.assert_allclose(loss, 20.415768)
    np.testing.assert_allclose(metrics["loss"].compute(), 20.415768)
    np.testing.assert_allclose(metrics["metrics/ndcg"].compute(), 0.41030282)
    np.testing.assert_allclose(metrics["metrics/mrr"].compute(), 0.30208334)


if __name__ == "__main__":
  absltest.main()
