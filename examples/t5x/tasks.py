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

"""Ranking tasks for T5X."""

from typing import Mapping

import seqio
import t5
import tensorflow as tf


def _msmarco_preprocessor(
    dataset: tf.data.Dataset,
    output_features: Mapping[str, t5.data.Feature],
    shuffle_lists: bool = True,
) -> tf.data.Dataset:
  """Preprocessor for MS-Marco listwise ranking task.

  Args:
    dataset: The ms_marco/v2.1 dataset.
    output_features: A mapping for each of the output features.
    shuffle_lists: If True, lists are shuffled which enforces uniformly random
      selection of documents when lists are truncated to a fixed list size.

  Returns:
    The processed dataset.
  """

  def extract_inputs(features):
    # Extract features from dataset and convert to model inputs.
    query = features["query"]
    passages = features["passages"]["passage_text"]
    inputs = tf.strings.join(
        ["Query:", query, "Document:", passages], separator=" "
    )
    label = tf.cast(features["passages"]["is_selected"], dtype=tf.float32)

    # The target is an unused token to obtain a ranking score.
    targets = tf.fill(tf.shape(label), "<extra_id_10>")

    # Shuffle lists, this enforces uniformly random selection of documents when
    # lists are later truncated to a fixed list size.
    if shuffle_lists:
      shuffle_idx = tf.random.shuffle(tf.range(tf.shape(label)[0]), seed=73)
      label = tf.gather(label, shuffle_idx)
      inputs = tf.gather(inputs, shuffle_idx)

    return {
        "inputs": inputs,
        "targets": targets,
        "label": label,
        "mask": tf.ones_like(label, dtype=tf.bool),
    }

  # Extract necessary inputs from MS-Marco dataset.
  dataset = dataset.map(extract_inputs, num_parallel_calls=tf.data.AUTOTUNE)

  # Tokenize only the text features, leave the others unchanged.
  tokenize_features = {
      "inputs": output_features["inputs"],
      "targets": output_features["targets"],
  }
  dataset = seqio.preprocessors.tokenize(dataset, tokenize_features)
  return dataset


_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(
        vocabulary=t5.data.get_default_vocabulary(),
        add_eos=False,
        required=False,
        rank=2,
    ),
    "targets": t5.data.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=False, rank=2
    ),
    "label": t5.data.Feature(
        vocabulary=seqio.PassThroughVocabulary(size=0),
        add_eos=False,
        required=False,
        dtype=tf.float32,
        rank=1,
    ),
    "mask": t5.data.Feature(
        vocabulary=seqio.PassThroughVocabulary(size=0),
        add_eos=False,
        required=False,
        dtype=tf.bool,
        rank=1,
    ),
}


t5.data.TaskRegistry.add(
    "msmarco_qna21_ranking",
    seqio.Task,
    source=seqio.TfdsDataSource(
        tfds_name="huggingface:ms_marco/v2.1", splits=["train", "validation"]
    ),
    preprocessors=[_msmarco_preprocessor],
    output_features=_OUTPUT_FEATURES,
)
