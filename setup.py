
import os
import subprocess
from setuptools import setup
from distutils.command.install import install as _install


class install(_install):
    def run(self):
        subprocess.call(['make', 'clean'])
        subprocess.call(['make', 'python'])
        _install.run(self)



setup(
    name='curig',  
    version='0.1',
    author="LIU Honghao",
    author_email="stein.h.liu@gmail.com",
    description="GPU version of NUFFT and Radio astronomy gridder package",
    packages=['curig'],
    package_dir={'curig': 'python/curagridder'},
    package_data={'curig': ['libcurafft.so']},
    url="https://github.com/HLSUD/CURIG",
    install_requires=['numpy', 'pycuda', 'six'],
    python_requires='>=3.6',
    zip_safe=False,
    include_package_data=True,
    cmdclass={'install': install},
)