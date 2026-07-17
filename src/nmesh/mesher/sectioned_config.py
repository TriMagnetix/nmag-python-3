"""Case-preserving, sectioned INI data used by meshing parameters."""

from __future__ import annotations

import configparser
from collections.abc import ItemsView
from io import StringIO
from os import PathLike
from typing import Any


class _CasePreservingConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return optionstr


class SectionedConfig:
    """Small mutable INI container with typed scalar coercion."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _coerce_value(value: str) -> Any:
        text = value.strip()
        if not text:
            return text
        if text.lower() in {"true", "false"}:
            return text.lower() == "true"
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            return text

    def _load_from_parser(self, parser: configparser.ConfigParser) -> None:
        for section in parser.sections():
            self.add_section(section)
            for name, value in parser.items(section):
                self.set(section, name, self._coerce_value(value))

    def from_file(self, file_path: str | PathLike[str]) -> None:
        parser = _CasePreservingConfigParser(delimiters=("=", ":"), interpolation=None)
        with open(file_path, encoding="utf-8") as stream:
            parser.read_file(stream)
        self._load_from_parser(parser)

    def from_string(self, string: str) -> None:
        parser = _CasePreservingConfigParser(delimiters=("=", ":"), interpolation=None)
        parser.read_file(StringIO(string))
        self._load_from_parser(parser)

    def add_section(self, section: str) -> None:
        self._data.setdefault(section, {})

    def get(self, section: str, name: str, raw: bool = False) -> Any:
        del raw
        return self._data.get(section, {}).get(name)

    def set(self, section: str, name: str, value: Any) -> None:
        self._data.setdefault(section, {})[name] = value

    def items(self, section: str) -> ItemsView[str, Any]:
        return self._data.get(section, {}).items()

    def to_string(self) -> str:
        return str(self._data)
