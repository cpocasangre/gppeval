
TOPIC
===============================
A Python stochastic library for assessing geothermal power potential by using the
volumetric method in a liquid-dominated reservoir

ABSTRACT
===============================
A Python stochastic library for assessing geothermal power potential by using
the volumetric method in a liquid dominated reservoir is presented in this 
research paper. More specifically, the purposes of this study are the use of the
volumetric method “heat in place” to estimate ability to produce electrical 
energy from geothermal liquid-dominated reservoir, and to code a valuable Python 
stochastic library that has the helpful methods for running the simulation. Even
though there are some kinds of licensed software for carrying out this simulation, 
for this task was selected Open-Source Programming Language, i.e., Python. The 
Geothermal Power Potential Evaluation stochastic library, GPPeval, is structured 
as three essential objects such as the geothermal power plant module, the Monte 
Carlo simulation module, and the module of tools.

Installation
============

Required Packages
-----------------

The following packages should be installed automatically (if using 'pip'
or 'easy_install'), otherwise they will need to be installed manually:

- NumPy_ : Numeric Python
- SciPy_ : Scientific Python
- Matplotlib_ : Python plotting library
- Mcerp : Monte Carlo Error Propagation

These packages come standard in *Python(x,y)*, *Spyder*, and other
scientific computing python bundles.

How to install
--------------

You have **several easy, convenient options** to install the 'gppeval'
package (administrative privileges may be required). Keep in mind to use Python 2.7

#. Simply copy the unzipped 'gppeval folder' directory to any other location that
   python can find it and rename it 'gppeval'.

#. From the command-line, do one of the following:

   a. Manually download the package files below, unzip to any directory, and
      run:

       $ [sudo] python setup.py install

   b. If 'setuptools' is installed, run:

       $ [sudo] easy_install [--upgrade] gppeval

   c. If 'pip' is installed, run:

       $ [sudo] pip install [--upgrade] gppeval
   
   d. If 'pip' is installed, run the command in the same folder:

       $ pip install [--upgrade] .

Contact
=======

Please send **feature requests, bug reports, or feedback** to:

'Carlos O. POCASANGRE JIMENEZ <carlos.pocasangre@mine.kyushu-u.ac.jp>'


.. _Monte Carlo methods: http://en.wikipedia.org/wiki/Monte_Carlo_method
.. _latin-hypercube sampling: http://en.wikipedia.org/wiki/Latin_hypercube_sampling
.. _soerp: http://pypi.python.org/pypi/soerp
.. _error propagation: http://en.wikipedia.org/wiki/Propagation_of_uncertainty
.. _math: http://docs.python.org/library/math.html
.. _NumPy: http://www.numpy.org/
.. _SciPy: http://scipy.org
.. _Matplotlib: http://matplotlib.org/
.. _scipy.stats: http://docs.scipy.org/doc/scipy/reference/stats.html
.. _uncertainties: http://pypi.python.org/pypi/uncertainties
.. _source code: https://github.com/tisimst/mcerp
.. _Abraham Lee: mailto:tisimst@gmail.com
.. _package documentation: http://pythonhosted.org/mcerp
.. _GitHub: http://github.com/tisimst/mcerp
.. _GitHub: http://github.com/cpocasangre/gppeval
