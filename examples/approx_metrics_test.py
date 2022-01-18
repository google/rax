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
"""Tests for rax.examples.approx_metrics."""

import io
from unittest import mock

from absl.testing import absltest
from jax import test_util as jtu

from examples import approx_metrics
import tensorflow_datasets as tfds

# The TFDS mock data generates float labels in [0, 1], which will always be 0
# for AP and R@50 metrics, so only NDCG is computed correctly in this test.
EXPECTED_OUTPUT = """             | AP      | NDCG    | R@50
ApproxAP     | 0.00000 | 0.82563 | 0.00000
ApproxNDCG   | 0.00000 | 0.83193 | 0.00000
ApproxR@50   | 0.00000 | 0.82563 | 0.00000
"""


class ApproxMetricsTest(jtu.JaxTestCase):

  def test_end_to_end(self):
    mock_stdout = io.StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      with tfds.testing.mock_data(
          num_examples=16, policy=tfds.testing.MockPolicy.USE_CODE):
        argv = ()
        approx_metrics.main(argv, epochs=1)

    # Get stdout output and remove trailing spaces:
    output = mock_stdout.getvalue()
    output = "\n".join([line.rstrip() for line in output.split("\n")])

    self.assertEqual(output, EXPECTED_OUTPUT)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
