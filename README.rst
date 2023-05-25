Benchmark for L1-Cox estimation
===============================
|Build Status| |Python 3.6+|


This benchmark is dedicated to solver of **L1-Cox estimation**:


$$
\min_{w} \sum_{i=1}^{n} -s_i \langle x_i, \beta \rangle + \log(\textstyle\sum_{y_j \geq y_i} e^{\langle x_j, \beta \rangle}) + \lambda \lVert w \rVert_1
$$


where $n$ (or ``n_samples``) stands for the number of samples, $p$ (or ``n_features``) stands for the number of features and


$$X \\in \\mathbb{R}^{n \\times p} \\ , \\quad w \\in \\mathbb{R}^p$$


Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/#ORG/#BENCHMARK_NAME
   $ benchopt run #BENCHMARK_NAME

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run #BENCHMARK_NAME -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Template| image:: https://github.com/benchopt/template_benchmark/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/template_benchmark/actions
.. |Build Status| image:: https://github.com/#ORG/#BENCHMARK_NAME/workflows/Tests/badge.svg
   :target: https://github.com/#ORG/#BENCHMARK_NAME/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
