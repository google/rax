.. title:: API

Ranking Losses (``rax.*_loss``)
===============================

.. automodule:: rax._src.losses
.. currentmodule:: rax

.. autosummary::

    pointwise_mse_loss
    pointwise_sigmoid_loss
    pairwise_hinge_loss
    pairwise_logistic_loss
    pairwise_mse_loss
    softmax_loss
    listmle_loss
    poly1_softmax_loss

.. autofunction:: pointwise_mse_loss
.. autofunction:: pointwise_sigmoid_loss
.. autofunction:: pairwise_hinge_loss
.. autofunction:: pairwise_logistic_loss
.. autofunction:: pairwise_mse_loss
.. autofunction:: softmax_loss
.. autofunction:: listmle_loss
.. autofunction:: poly1_softmax_loss

Ranking Metrics (``rax.*_metric``)
==================================

.. automodule:: rax._src.metrics
.. currentmodule:: rax

.. autosummary::

    mrr_metric
    precision_metric
    recall_metric
    ap_metric
    dcg_metric
    ndcg_metric

.. autofunction:: mrr_metric
.. autofunction:: precision_metric
.. autofunction:: recall_metric
.. autofunction:: ap_metric
.. autofunction:: dcg_metric
.. autofunction:: ndcg_metric

Function Transformations (``rax.*_t12n``)
=========================================

.. automodule:: rax._src.t12n
.. currentmodule:: rax

.. autosummary::

    approx_t12n
    bound_t12n
    gumbel_t12n

.. autofunction:: approx_t12n
.. autofunction:: bound_t12n
.. autofunction:: gumbel_t12n

Utilities
=========

.. currentmodule:: rax.utils

.. autosummary::

    ranks
    cutoff
    approx_ranks
    approx_cutoff

.. autofunction:: ranks
.. autofunction:: cutoff
.. autofunction:: approx_ranks
.. autofunction:: approx_cutoff

Types
=====

.. automodule:: rax._src.types
.. currentmodule:: rax.types

.. autosummary::

    CutoffFn
    LossFn
    MetricFn
    RankFn
    ReduceFn

.. autoclass:: CutoffFn
   :members: __call__

.. autoclass:: LossFn
   :members: __call__

.. autoclass:: MetricFn
   :members: __call__

.. autoclass:: RankFn
   :members: __call__

.. autoclass:: ReduceFn
   :members: __call__

References
==========

.. bibliography:: references.bib

