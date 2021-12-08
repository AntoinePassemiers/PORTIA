# -*- coding: utf-8 -*-
# setup.py
# author: Antoine Passemiers

from setuptools import setup


packages = [
    'portia',
    'portia.evaluation',
    'portia.optim'
]


setup(
    name='portia',
    version='1.0.0',
    description='Fast and Accurate Inference of Gene Regulatory Networks through Robust Precision Matrix Estimation',
    author='Antoine Passemiers',
    packages=packages,
    package_data={})
