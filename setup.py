import os
from setuptools import setup, find_packages
import gppeval


def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()


readme = 'README.rst'
license = 'LICENSE'

setup(
    name=gppeval.__module_name__,
    version=gppeval.__version__,
    description=gppeval.__description__,
    url=gppeval.__url__,
    author=gppeval.__author__,
    author_email=gppeval.__author_email__,
    license=gppeval.__license__,
    long_description=read(readme),
    packages=find_packages(exclude=['test*']),
    keywords=['monte carlo latin hypercube geothermal power potential volumetric method geothermal reservoir'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'mcerp', 'beautifultable'],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 2.7'],
    zip_safe=False,
    package_data={
        'gppeval': ['example/example.ipynb', 'example/reservoir_properties_list.csv'],
    },
)
