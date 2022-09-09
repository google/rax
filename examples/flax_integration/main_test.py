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

# pytype: skip-file
"""Tests for rax.examples.flax_integration."""

import io
import json
from unittest import mock

from absl.testing import absltest
import numpy as np

from examples.flax_integration import main
import tensorflow_datasets as tfds


class Web30kTest(absltest.TestCase):

  def test_end_to_end(self):
    mock_stdout = io.StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      with tfds.testing.mock_data(
          num_examples=16, policy=tfds.testing.MockPolicy.USE_CODE):
        argv = ()
        main.main(argv)

    # Get stdout output and parse json.
    output = json.loads(mock_stdout.getvalue())

    # Epochs should increase.
    self.assertEqual(output[0]["epoch"], 1)
    self.assertEqual(output[1]["epoch"], 2)
    self.assertEqual(output[2]["epoch"], 3)

    # Loss should decrease consistently.
    self.assertGreater(output[0]["loss"], output[1]["loss"])
    self.assertGreater(output[1]["loss"], output[2]["loss"])

    # Metrics should increase consistently.
    self.assertLess(output[0]["metric/ndcg"], output[1]["metric/ndcg"])
    self.assertLess(output[1]["metric/ndcg"], output[2]["metric/ndcg"])
    self.assertLess(output[0]["metric/ndcg@10"], output[1]["metric/ndcg@10"])
    self.assertLess(output[1]["metric/ndcg@10"], output[2]["metric/ndcg@10"])

    # Evaluate metric values after training.
    np.testing.assert_allclose(output[2]["metric/ndcg"], 0.829134, atol=0.02)
    np.testing.assert_allclose(output[2]["metric/ndcg@10"], 0.650672, atol=0.02)


if __name__ == "__main__":
  absltest.main()
