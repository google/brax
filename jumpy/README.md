# Jumpy

Jumpy is a common backend for [JAX](https://github.com/google/jax) or
[numpy](https://numpy.org/):

* A jumpy function returns a JAX outputs if given a JAX inputs
* A jumpy function returns a JAX outputs if jitted
* Otherwise a jumpy function returns numpy outputs

Jumpy lets you write framework agnostic code that is easy to debug by running
as raw numpy, but is just as performant as JAX when jitted.

## Installing Jumpy

To install Jumpy from pypi:

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install jumpy
```

Alternatively, to install Jumpy from source, clone this repo, `cd` to it, and then:

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
cd jumpy
pip install -e .
```
