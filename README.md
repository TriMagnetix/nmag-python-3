# NMAG Python 3

This is an in-progress port of [nmag](https://github.com/nmag-project/nmag-src) that updates old dependencies and packages nmag as a standalone python 3 module.

The Python 3 port is intended to be standalone and must not require OCaml at runtime. See [PARITY_COMPLETION.md](PARITY_COMPLETION.md) for the Section 4 parity record and validation gate.

### Installation
* Create a virtual environment: `python -m venv venv`
* Activate the virtual environment:
  * On Windows: `venv\Scripts\activate`
  * On macOS/Linux: `source venv/bin/activate`
* To initialize the project and install dependencies: `pip install -e .`

### Testing
* To install the optional test dependencies run `pip install -e ".[test]"`
* Once the project has been initialized simply run `pytest` to run your tests
* To generate modern nmesh parity scenario outputs run `python tools/nmesh_parity_compare.py --output-dir parity/new`
* To compare against legacy `.nmesh` artifacts run `python tools/nmesh_parity_compare.py --legacy-dir parity/legacy`
* To view coverage open `htmlcov/index.html` in your browser
* [Specifying what tests to run](https://docs.pytest.org/en/latest/how-to/usage.html#specifying-which-tests-to-run)
