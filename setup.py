import os
from setuptools import setup, find_packages
import gppeval


def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()

readme = 'README.rst'

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
    # include_package_data=True,
    keywords=['monte carlo latin hypercube geothermal power potential volumetric method geothermal reservoir'],
    install_requires=['numpy', 
                      'scipy', 
                      'matplotlib', 
                      'mcerp', 
                      'beautifultable', 
                      'iapws'],
    classifiers=['Development Status :: 4 - Beta',
                 'Environment :: Console',
                 'Environment :: MacOS X',
                 'Environment :: X11 Applications',
                 'Environment :: Win32 (MS Windows)',
                 'Natural Language :: English',
                 'Operating System :: Microsoft',
                 'Operating System :: POSIX :: Linux',
                 'Operating System :: MacOS :: MacOS X',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.8'],
    zip_safe=False,
    #package_data={
    #    'gppeval': ['example/example.ipynb',
    #                'example/reservoir_properties_list.csv',
    #                'example/testGppeval.py'],
    #},
)
