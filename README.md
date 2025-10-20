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
