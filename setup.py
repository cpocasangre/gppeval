import os
from setuptools import setup, find_packages


def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()


readme = 'README.rst'
license = 'LICENSE'

setup(
    name='gppeval',
    version='2018.1.22.0.1',
    description='Geothermal Power Potential assessment',
    url='https://github.com/cpocasangre/gppeval',
    author='Carlos O. POCASANGRE JIMENEZ',
    author_email='carlos.pocasangre@mine.kyushu-u.ac.jp',
    license='MIT License',
    long_description=read(readme),
    packages=find_packages(exclude=['test*']),
    keywords=['monte carlo', 'latin hypercube', 'geothermal power potential'],
    classifiers=[
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2.7',
    ],)
