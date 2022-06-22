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
import json
from unittest import mock

from absl.testing import absltest
import numpy as np

from examples.approx_metrics import main
import tensorflow_datasets as tfds


class ApproxMetricsTest(absltest.TestCase):

  def test_end_to_end(self):
    mock_stdout = io.StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      with tfds.testing.mock_data(
          num_examples=16, policy=tfds.testing.MockPolicy.USE_CODE):
        argv = ()
        main.main(argv, epochs=1)

    # Get stdout output and parse json.
    output = json.loads(mock_stdout.getvalue())

    # The TFDS mock data generates float labels in [0, 1], which will always be
    # 0 for AP and R@50 metrics, so only NDCG is computed correctly and tested.
    np.testing.assert_allclose(output["ApproxAP"]["NDCG"], 0.82563, rtol=0.01)
    np.testing.assert_allclose(output["ApproxNDCG"]["NDCG"], 0.83193, rtol=0.01)
    np.testing.assert_allclose(output["ApproxR@50"]["NDCG"], 0.82563, rtol=0.01)


if __name__ == "__main__":
  absltest.main()
