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

# pytype: skip-file
"""Tests for rax.examples.segmentation."""

import io
import json
from unittest import mock

from absl.testing import absltest
import jax
import numpy as np

from examples.segmentation import main
import tensorflow_datasets as tfds

# Opt-in to the partitionable PRNG implementation.
jax.config.update("jax_threefry_partitionable", True)


class SegmentationTest(absltest.TestCase):

  def test_end_to_end(self):
    mock_stdout = io.StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      with tfds.testing.mock_data(
          num_examples=16, policy=tfds.testing.MockPolicy.USE_CODE
      ):
        argv = ()
        main.main(argv, steps=60, steps_per_eval=20)

    # Get stdout output and parse json.
    output = json.loads(mock_stdout.getvalue())

    # Epochs should increase.
    with self.subTest(name="Steps increase"):
      self.assertEqual(output[0]["step"], 20)
      self.assertEqual(output[1]["step"], 40)
      self.assertEqual(output[2]["step"], 60)

    # Loss should decrease consistently.
    with self.subTest("Loss decreases consistently"):
      self.assertGreater(output[0]["loss"], output[1]["loss"])
      self.assertGreater(output[1]["loss"], output[2]["loss"])

    # NDCG@10 metric should increase consistently.
    with self.subTest(name="NDCG@10 increases consistently"):
      self.assertLess(output[0]["ndcg@10"], output[1]["ndcg@10"])
      self.assertLess(output[1]["ndcg@10"], output[2]["ndcg@10"])

    # Evaluate exact NDCG@10 metric value after training.
    with self.subTest(name="Exact NDCG@10 value after training"):
      np.testing.assert_allclose(output[2]["ndcg@10"], 0.732969, atol=0.03)


if __name__ == "__main__":
  absltest.main()
