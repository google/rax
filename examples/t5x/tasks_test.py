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
"""Tests for rax.examples.t5x.tasks."""

from unittest import mock

from absl.testing import absltest
from examples.t5x import tasks
import tensorflow as tf


class TasksTest(tf.test.TestCase):

  def test_msmarco_preprocessor(self):
    # Input sample data.
    ds = {
        "query": tf.constant(b"who founded google?"),
        "passages": {
            "passage_text":
                tf.constant([
                    b"larry page and sergey brin",
                    b"google is an American multinational technology company",
                    b"google was founded in 1998",
                ]),
            "is_selected":
                tf.constant([1, 0, 0])
        }
    }
    ds = tf.data.Dataset.from_tensors(ds)

    # Perform preprocessing with mocked features.
    tf.random.set_seed(42)
    inputs_mock = mock.Mock()
    inputs_mock.vocabulary.encode_tf.return_value = tf.ragged.constant(
        [[5, 1024, 250, 1, 24, 205, 555], [5, 304, 200, 200],
         [1, 204, 333, 1024, 5]],
        dtype=tf.int32)
    targets_mock = mock.Mock()
    targets_mock.vocabulary.encode_tf.return_value = tf.ragged.constant(
        [[10], [10], [10]], dtype=tf.int32)
    label_mock = mock.Mock()
    mask_mock = mock.Mock()
    ds = tasks._msmarco_preprocessor(
        ds, {
            "inputs": inputs_mock,
            "targets": targets_mock,
            "label": label_mock,
            "mask": mask_mock
        })

    # Get a single batch of data and validate its contents.
    batch = next(iter(ds))
    expected = {
        "inputs_pretokenized":
            tf.constant([
                b"Query: who founded google? Document: google was founded in 1998",
                b"Query: who founded google? Document: larry page and sergey brin",
                b"Query: who founded google? Document: google is an American multinational technology company"
            ]),
        "inputs":
            tf.ragged.constant([[5, 1024, 250, 1, 24, 205, 555],
                                [5, 304, 200, 200], [1, 204, 333, 1024, 5]],
                               dtype=tf.int32),
        "targets_pretokenized":
            tf.constant([b"<extra_id_10>", b"<extra_id_10>", b"<extra_id_10>"]),
        "targets":
            tf.ragged.constant([[10], [10], [10]], dtype=tf.int32),
        "label":
            tf.constant([0., 1., 0.], dtype=tf.float32),
        "mask":
            tf.constant([True, True, True], dtype=tf.bool)
    }

    for key in batch:
      self.assertAllEqual(batch[key], expected[key])


if __name__ == "__main__":
  absltest.main()
