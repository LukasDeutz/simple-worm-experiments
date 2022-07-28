'''
Created on 28 Jun 2022

@author: lukas
'''
import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = 'simple_worm_experiments',
    version = '0.1',
    author = 'Lukas Deutz',
    author_email = 'scld@leeds.ac.uk',
    description = 'Module to run simple-worm forward experiments',
    long_description = read('README.md'),
    url = 'https://github.com/LukasDeutz/simple-worm-experiments.git',
    packages = find_packages()
)


