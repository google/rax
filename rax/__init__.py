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

"""Rax: Learning to Rank using JAX."""

from rax._src.losses import mse_loss
from rax._src.losses import pairwise_hinge_loss
from rax._src.losses import pairwise_logistic_loss
from rax._src.losses import pairwise_mse_loss
from rax._src.losses import sigmoid_cross_entropy_loss
from rax._src.losses import softmax_loss
from rax._src.metrics import ap_metric
from rax._src.metrics import dcg_metric
from rax._src.metrics import mrr_metric
from rax._src.metrics import ndcg_metric
from rax._src.metrics import precision_metric
from rax._src.metrics import recall_metric
from rax._src.t12n import approx_t12n
from rax._src.t12n import bound_t12n
from rax._src.t12n import gumbel_t12n
from rax._src.utils import approx_ranks
from rax._src.utils import sort_ranks

__version__ = "0.0.1"

# pyformat: disable
__all__ = [
    "mse_loss",
    "pairwise_hinge_loss",
    "pairwise_logistic_loss",
    "pairwise_mse_loss",
    "sigmoid_cross_entropy_loss",
    "softmax_loss",
    "ap_metric",
    "dcg_metric",
    "mrr_metric",
    "ndcg_metric",
    "precision_metric",
    "recall_metric",
    "approx_t12n",
    "bound_t12n",
    "gumbel_t12n",
    "approx_ranks",
    "sort_ranks"
]
# pyformat: enable

# copybara: stripped(1)

#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the Rax public API.     /
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
