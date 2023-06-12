import fnmatch
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as build_py_orig
from setuptools.command.install import install as _install
import os
from pkg_resources import resource_filename
import subprocess
from .cfg import *
import logging

class OverrideInstall(_install):
    def run(self):
        print(bash_path)
        subprocess.run(["bash",f"{bash_path}"])    
        _install.run(self)
    
excluded = ['Tests/*.ipynb']

class build_py(build_py_orig):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            (pkg, mod, file)
            for (pkg, mod, file) in modules
            if not any(fnmatch.fnmatchcase(file, pat=pattern) for pattern in excluded)
        ]
    
setup(name='covid_dataset',
version='1.0',
description='A package for getting covid_dataset network in Brazil',
url='#',
author='Felipe Schreiber Fernandes',
install_requires=[""],
author_email='felipesc@cos.ufrj.br',
packages=find_packages(),
cmdclass={'build_py': build_py,"install":OverrideInstall},
zip_safe=False)