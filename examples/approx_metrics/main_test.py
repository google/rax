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

# pytype: skip-file
"""Tests for rax.examples.approx_metrics."""

import io
import json
from unittest import mock

from absl.testing import absltest
import numpy as np

from examples.approx_metrics import main
import tensorflow as tf
import tensorflow_datasets as tfds


def letor_dataset(self, *args, **kwargs):
  del args, kwargs  # Unused but needed for `tfds.testing.mock_data()`.
  num_features = 136
  num_examples = 2

  def generator():
    rng = np.random.default_rng(42)
    list_sizes = np.int32(rng.uniform(30, 80, size=(num_examples,)))
    for list_size in list_sizes:
      features = np.float64(rng.normal(size=(list_size, num_features)))
      labels = np.float64(rng.binomial(2, 0.2, size=(list_size,)))
      yield {
          "label": labels,
          "float_features": features,
          "query_id": np.array("qid"),
          "doc_id": np.arange(list_size),
      }

  return tf.data.Dataset.from_generator(
      generator,
      output_types=self.info.features.dtype,
      output_shapes=self.info.features.shape,
  )


class ApproxMetricsTest(absltest.TestCase):

  def test_end_to_end(self):
    mock_stdout = io.StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      with tfds.testing.mock_data(as_dataset_fn=letor_dataset):
        argv = ()
        main.main(argv, epochs=10)

    # Get stdout output and parse json.
    output = json.loads(mock_stdout.getvalue())

    # ApproxAP works best on AP.
    np.testing.assert_allclose(output["ApproxAP"]["AP"], 0.88122, rtol=1e-3)
    np.testing.assert_allclose(output["ApproxNDCG"]["AP"], 0.70764, rtol=1e-3)
    np.testing.assert_allclose(output["ApproxR@50"]["AP"], 0.45507, rtol=1e-3)

    # ApproxNDCG works best on NDCG
    np.testing.assert_allclose(output["ApproxAP"]["NDCG"], 0.79133, rtol=1e-3)
    np.testing.assert_allclose(output["ApproxNDCG"]["NDCG"], 0.90464, rtol=1e-3)
    np.testing.assert_allclose(output["ApproxR@50"]["NDCG"], 0.67759, rtol=1e-3)

    # ApproxR@50 is not best on R@50 due to difficulty of that metric.
    np.testing.assert_allclose(output["ApproxNDCG"]["R@50"], 0.92857, rtol=1e-3)
    np.testing.assert_allclose(output["ApproxAP"]["R@50"], 0.92460, rtol=1e-3)
    np.testing.assert_allclose(output["ApproxR@50"]["R@50"], 0.85317, rtol=1e-3)


if __name__ == "__main__":
  absltest.main()
