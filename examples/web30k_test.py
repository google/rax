# pytype: skip-file
"""Tests for rax.examples.web30k."""

import io
from unittest import mock

from absl.testing import absltest
from jax import test_util as jtu

from examples import web30k
import tensorflow_datasets as tfds

EXPECTED_OUTPUT = """epoch=1
  loss=3.14508
  metric/mrr=0.00000
  metric/ndcg=0.81562
  metric/ndcg@10=0.62021
epoch=2
  loss=3.14373
  metric/mrr=0.00000
  metric/ndcg=0.82403
  metric/ndcg@10=0.63671
epoch=3
  loss=3.14263
  metric/mrr=0.00000
  metric/ndcg=0.82581
  metric/ndcg@10=0.63811
"""


class Web30kTest(jtu.JaxTestCase):

  def test_end_to_end(self):
    mock_stdout = io.StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      with tfds.testing.mock_data(
          num_examples=256, policy=tfds.testing.MockPolicy.USE_CODE):
        argv = ()
        web30k.main(argv)

    output = mock_stdout.getvalue()
    self.assertEqual(output, EXPECTED_OUTPUT)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
