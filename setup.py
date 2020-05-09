
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy
import scipy
extensions = [
    Extension("main", ["main.pyx"],
				include_dirs=[numpy.get_include(), scipy.get_include()],
				extra_compile_args=['-O2', '-march=native', '/openmp'],
				extra_link_args=['-O2', '-march=native', '/openmp'],) ]
setup(
    name= "main_cython",
    ext_modules=cythonize(extensions, annotate = True,),
)
