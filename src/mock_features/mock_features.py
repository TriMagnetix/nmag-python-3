import configparser
from io import StringIO
from typing import Any, Dict

class MockFeatures:
    """
    A unified stub to replace nsim.features.Features and provide
    default configuration values for simulations and nmesh.
    """
    def __init__(self):
        # Default configuration for simulations
        self._data: Dict[str, Dict[str, Any]] = {
            'etc': {
                'runid': 'nmag_simulation',  # Default output filename
                'savedir': '.',              # Default output directory
            },
            'nmag': {
                'clean': False,              # Don't delete old files automatically
                'restart': False,            # Don't try to restart from h5
            }
        }

    @staticmethod
    def _coerce_value(value: str) -> Any:
        text = value.strip()
        if text == "":
            return text

        lowered = text.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"

        try:
            return int(text)
        except ValueError:
            pass

        try:
            return float(text)
        except ValueError:
            return text

    def _load_from_parser(self, parser: configparser.ConfigParser):
        for section in parser.sections():
            self.add_section(section)
            for name, value in parser.items(section):
                self.set(section, name, self._coerce_value(value))

    def from_file(self, file_path):
        """Loads INI-style features from a file."""
        parser = configparser.ConfigParser(
            delimiters=("=", ":"),
            interpolation=None,
        )
        parser.optionxform = str
        with open(file_path, encoding="utf-8") as stream:
            parser.read_file(stream)
        self._load_from_parser(parser)

    def from_string(self, string):
        """Loads INI-style features from a string."""
        parser = configparser.ConfigParser(
            delimiters=("=", ":"),
            interpolation=None,
        )
        parser.optionxform = str
        parser.read_file(StringIO(string))
        self._load_from_parser(parser)

    def add_section(self, section: str):
        """Adds a section to the features if it doesn't exist."""
        if section not in self._data:
            self._data[section] = {}

    def get(self, section: str, name: str, raw: bool = False) -> Any:
        """Retrieves a value from the specified section and name."""
        return self._data.get(section, {}).get(name)

    def set(self, section: str, name: str, value: Any):
        """Sets a value in the specified section and name."""
        if section not in self._data:
            self._data[section] = {}
        self._data[section][name] = value

    def items(self, section: str):
        """Returns items in a given section."""
        return self._data.get(section, {}).items()

    def to_string(self) -> str:
        """Returns string representation of the data."""
        return str(self._data)
