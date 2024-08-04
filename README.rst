TOPIC
===============================
A Python-based stochastic library for assessing geothermal power potential using the volumetric
method in a liquid-dominated reservoir.

Authors
--------------
- Carlos Pocasangre Jiménez (carlos.pocasangre@ues.edu.sv)

- Fidel Ernesto Cortez Torres (ernestocortez.sv@ieee.org)

- Rubén Alexander Henríquez Miranda (rubenhenriquez@ieee.org)

ABSTRACT
===============================
We present a Python-based stochastic library for assessing geothermal power
potential using the volumetric method in a liquid-dominated reservoir.
The specific aims of this study are to use the volumetric method, “heat in
place,” to estimate electrical energy production ability from a geothermal
liquid-dominated reservoir, and to build a Python-based stochastic library
with useful methods for running such simulations. Although licensed
software is available, we selected the open-source programming language
Python for this task. The Geothermal Power Potential Evaluation stochastic
library (*gppeval*) is structured as three essential objects including a
geothermal power plant module, a Monte Carlo simulation module, and a tools
module.

For testing the application, a **Jupyter Notebook** example has been included in the `example
folder`_.

*HINT*: **Now, this application is available for Python 3.5**

Reference
--------------
Pocasangre, C., & Fujimitsu, Y. (2018). *A Python-based stochastic library for assessing
geothermal power potential using the volumetric method in a liquid-dominated reservoir*.
**Geothermics**, 76, 164-176.
https://doi.org/10.1016/J.GEOTHERMICS.2018.07.009

J. Lawless. 2010. Geothermal Lexicon For Resources and Reserves Definition
and Reporting. 2nd Edition (2010) Edition. Adelaide, Southern Australia:
Australian Geothermal Reporting Code Committee (AGRCC)

INSTALLATION
============

Required Packages
-----------------

The following packages should be installed automatically (if using 'pip'
or 'easy_install'), otherwise they will need to be installed manually:

- NumPy_ : Numeric Python
- SciPy_ : Scientific Python
- Matplotlib_ : Python plotting library
- Mcerp_ : Monte Carlo Error Propagation
- Iapws_ : The InternationalAssociation for the Properties of Water and Steam
- Beautifultable_ : Utility package to print visually appealing ASCII tables to terminal

How to install
--------------

You have **several easy, convenient options** to install the 'gppeval'
package (administrative privileges may be required).

#. Simply copy the unzipped 'gppeval folder' directory to any other location that
   python can find it and rename it 'gppeval'.

#. From the command-line, do one of the following:

   a. Manually download the package files below, unzip to any directory, and
      run:

       $ [sudo] python setup.py install

   b. If 'pip' is installed, run the follow command (stable version and internet connection is required)

       $ [sudo] pip install [--upgrade] gppeval

CHANGES OF NEW ISSUE
====================

#. gppeval (2024.08.04.0.1.dev1).
    Fixed bugs.

#. gppeval (2020.10.1.0.3.dev1).
    Added tho-phases reservoir equation.
    Fixed bugs.

#. gppeval (2019.4.17.0.6.dev1).
    Python 3.8
    Fixed bugs.

#. gppeval (2019.4.17.0.2.dev1).
    Python 3.5 available

#. gppeval (2018.10.11.0.1.dev1).
    The input file csv has been modified. It includes the possibility of using volume as a input
    reservoir parameter. Using the word ``none`` is possible to exchange between either to use
    **Area** and **Thickness** or to use only **Volume** as a reservoir geometric parameter.

    Example: Using Area and Thickness

        0,Name,14.00061,-88.73744,ReservoirArea,A,km2,5,6,7,0,0,T
        1,,,,Thickness,h,m,450,500,600,0,0,T
        2,,,,Volume,v,km3,4,6,8.2,0,0,none

    Example: Using only Volume

        0,Name,14.00061,-88.73744,ReservoirArea,A,km2,5,6,7,0,0,None
        1,,,,Thickness,h,m,450,500,600,0,0,None
        2,,,,Volume,v,km3,4,6,8.2,0,0,T

#. gppeval (2018.4.6.0.1.dev1).
    Original issue after have been upload as a stable.

#. gppeval (2017.10.1.0.1.dev1).
    Original issue.

CONTACT
=======

Please send **feature requests, bug reports, or feedback** to: `Carlos O. POCASANGRE JIMENEZ`_

.. _Monte Carlo methods: http://en.wikipedia.org/wiki/Monte_Carlo_method
.. _latin-hypercube sampling: http://en.wikipedia.org/wiki/Latin_hypercube_sampling
.. _error propagation: http://en.wikipedia.org/wiki/Propagation_of_uncertainty
.. _math: http://docs.python.org/library/math.html
.. _NumPy: http://www.numpy.org/
.. _SciPy: http://scipy.org
.. _Matplotlib: http://matplotlib.org/
.. _scipy.stats: http://docs.scipy.org/doc/scipy/reference/stats.html
.. _uncertainties: http://pypi.python.org/pypi/uncertainties
.. _Mcerp: http://github.com/tisimst/mcerp
.. _Beautifultable: https://github.com/pri22296/beautifultable
.. _Gppeval: http://github.com/cpocasangre/gppeval
.. _example folder: https://github.com/cpocasangre/gppeval
.. _Carlos O. POCASANGRE JIMENEZ: mailto:carlos.pocasangre@ues.edu.sv
.. _Iapws: https://pypi.org/project/iapws/
