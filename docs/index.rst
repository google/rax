:github_url: https://github.com/google/rax/tree/main/docs

🦖 Rax: Learning-to-Rank using JAX
==================================

.. toctree::
   :hidden:
   :maxdepth: 0

   🦖 Rax Documentation <self>

Rax is a Learning-to-Rank (LTR) library built on top of JAX.

.. code-block:: python

    import rax
    import jax.numpy as jnp

    scores = jnp.array([0.3, 0.8, 0.21])
    labels = jnp.array([1., 0., 2.])

    rax.pairwise_hinge_loss(scores, labels)
    rax.ndcg_metric(scores, labels)


Installation
------------

See https://github.com/jax-ml/jax#pip-installation for instructions on
installing JAX.

We suggest installing the latest stable version of Rax by running::

    $ pip install rax


.. toctree::
   :caption: API Documentation
   :maxdepth: 1

   api

.. toctree::
   :caption: Examples
   :maxdepth: 1
   :glob:

   examples/*

Contribute
----------

- Issue tracker: https://github.com/google/rax/issues
- Source code: https://github.com/google/rax/tree/master

Support
-------

If you are having issues, please let us know by filing an issue on our
`issue tracker <https://github.com/google/rax/issues>`_.

License
-------

Rax is licensed under the Apache 2.0 License.

