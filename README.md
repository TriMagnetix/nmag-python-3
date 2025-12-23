# NMAG Python 3

This is an in-progress port of [nmag](https://github.com/nmag-project/nmag-src) that updates old dependencies and packages nmag as a standalone python 3 module.

### Installation
* Create a virtual environment: `python -m venv venv`
* Activate the virtual environment:
  * On Windows: `venv\Scripts\activate`
  * On macOS/Linux: `source venv/bin/activate`
* To initialize the project and install dependencies: `pip install -e .`

### Testing
* To install the optional test dependencies run `pip install -e ".[test]"`
* Once the project has been initialized simply run `pytest` to run your tests
* To view coverage open `htmlcov/index.html` in your browser
* [Specifying what tests to run](https://docs.pytest.org/en/latest/how-to/usage.html#specifying-which-tests-to-run)

### OCaml
* Install latest Ocaml/Opam version 4 https://ocaml.org/install
  * Recommended to run `opam init` and setup a switch with version 4
* Install ocaml-in-python `opam install ocaml-in-python`
* Register the package in python ``pip install --editable "`opam var ocaml-in-python:lib`"``
* Tell python where to look for the OCaml library `export OCAMLPATH=${DUNE_DIR}/_build/install/default/lib` where DUNE_DIR is the directory of the dune project
* Run this to explictly activate the Opam switch `eval $(opam env)`
