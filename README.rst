Benchmark for Cox estimation
===============================
|Build Status| |Python 3.6+|


This benchmark is dedicated to solver of **Cox estimation**:


$$
\\min_{w} \\frac{1}{n} \\sum_{i=1}^{n} -s_i \\langle x_i, w \\rangle + \\log(\\textstyle\\sum_{y_j \\geq y_i} e^{\\langle x_j, w \\rangle})
+ \\lambda \\Big( \\rho \\lVert w \\rVert_1 + \\frac{1-\\rho}{2} \\lVert w \\rVert^2_2 \\Big)
$$

where $n$ (or ``n_samples``) stands for the number of samples, $p$ (or ``n_features``) stands for the number of features, $s$ the vector of observation censorship, $y$ occurrences times.


$$\\mathbf{X} \\in \\mathbb{R}^{n \\times p} \\ , \\, s \\in \\{ 0, 1 \\}^n, \\ y \\in \\mathbb{R}^n, \\quad w \\in \\mathbb{R}^p$$


In the case of tied data, data with observation having the same occurrences time, the objective reads

$$
\\min_{w} \\frac{1}{n} \\sum_{l=1}^{m} \\bigg(
\\sum_{i \\in H_{i_l}} - \\langle x_i, w \\rangle 
+ \\log \\Bigl(\\textstyle \\sum_{y_j \\geq y_{i_l}} e^{\\langle x_j, w \\rangle} - \\frac{\\#(i) - 1}{\\lvert H_{i_l} \\rvert}\\textstyle\\sum_{j \\in H_{i_l}} e^{\\langle x_j, w \\rangle}\\Bigl)
\\bigg)
+ \\lambda \\Big( \\rho \\lVert w \\rVert_1 + \\frac{1-\\rho}{2} \\lVert w \\rVert^2_2 \\Big)
$$

where $H_{i_l} = \\{ i \\ | \\ s_i = 1 \\ ;\\ y_{i} = y_{i_l} \\}$ is the set of uncensored observations with same occurrence time $y_{i_l}$ and $\\#(i)$ the index of observation $i$ in $H_{i_l}$.


Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_l1_cox
   $ cd benchmark_l1_cox
   $ benchopt run .

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run . -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.


.. |Build Status| image:: https://github.com/benchopt/benchmark_l1_cox/workflows/Tests/badge.svg
   :target: https://github.com/#ORG/#BENCHMARK_NAME/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
