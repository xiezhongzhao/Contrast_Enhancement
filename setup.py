#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: xiezhongzhao@cug.edu.cn


import setuptools
from setuptools import setup

setup(
    name='ContrastEnhancement',
    version='0.0.1',
    description='image contrast enhancement',
    author='xiezhongzhao',
    author_email='xiezhongzhao@cug.edu.cn',
    license='MIT',
    url='xiezhongzhao.github.io',
    packages=setuptools.find_packages(),

    install_requires=[
        'numpy',
        'opencv-python',
        'matplotlib',
        'scipy',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: Microsoft'  
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries'
    ],
    platforms = "any",
    include_package_data=False,
    zip_safe=True,
)
# python setup.py sdist bdist_wheel --universal