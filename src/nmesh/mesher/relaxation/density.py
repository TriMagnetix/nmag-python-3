"""Density-function parsing and compilation helpers."""

from __future__ import annotations

import math
import re

import numpy as np

from ._constants import DENSITY_EPSILON
from ._types import DensityFunction, FloatArray


def _strip_c_comments(source: str) -> str:
    """Remove C-style block comments from a legacy density script."""

    return re.sub(r"/\*.*?\*/", "", source, flags=re.S)


def _translate_density_source(source: str) -> str:
    """Translate a small subset of legacy C-like density syntax into Python."""

    cleaned = _strip_c_comments(source)
    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = cleaned.replace("{", "\n{\n").replace("}", "\n}\n").replace(";", ";\n")
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    py_lines = ["def __compiled_density(x):", "    density = 1.0"]
    indent = 1

    for line in lines:
        if line == "{":
            indent += 1
            continue
        if line == "}":
            indent = max(1, indent - 1)
            continue

        translated = line.rstrip(";").strip()
        translated = re.sub(r"\bdouble\s+", "", translated)
        translated = translated.replace("&&", " and ").replace("||", " or ")
        translated = re.sub(r"(?<![<>=!])!(?!=)", " not ", translated)
        translated = re.sub(r"\bexp\s*\(", "math.exp(", translated)
        translated = re.sub(r"\bsqrt\s*\(", "math.sqrt(", translated)
        translated = re.sub(r"\bsin\s*\(", "math.sin(", translated)
        translated = re.sub(r"\bcos\s*\(", "math.cos(", translated)
        translated = re.sub(r"\btan\s*\(", "math.tan(", translated)
        translated = re.sub(r"\bpow\s*\(", "math.pow(", translated)

        if translated.startswith("if "):
            condition = translated[2:].strip()
            if condition.startswith("(") and condition.endswith(")"):
                condition = condition[1:-1].strip()
            py_lines.append(f"{'    ' * indent}if {condition}:")
            continue

        if translated == "else":
            py_lines.append(f"{'    ' * indent}else:")
            continue

        py_lines.append(f"{'    ' * indent}{translated}")

    py_lines.append("    return float(density)")
    return "\n".join(py_lines)


def _compile_density_function(density: str | DensityFunction | None) -> DensityFunction:
    """Compile a legacy density string or callable into a normalized density function."""

    if density is None or density == "":
        return lambda _point: 1.0

    if callable(density):

        def density_wrapper(point: FloatArray) -> float:
            """Wrap a user density callable with array coercion and clamping."""

            value = float(density(np.asarray(point, dtype=float)))
            return max(value, DENSITY_EPSILON)

        return density_wrapper

    source = str(density).strip()
    try:
        expression = re.search(r"density\s*=\s*(.+?)\s*;", source, flags=re.S)
        if expression is not None and "if" not in source and "{" not in source:
            code = compile(expression.group(1), "<density>", "eval")

            def expression_density(point: FloatArray) -> float:
                """Evaluate a single-expression density snippet."""

                x = np.asarray(point, dtype=float)
                scope = {
                    "x": x,
                    "math": math,
                    "exp": math.exp,
                    "sqrt": math.sqrt,
                    "sin": math.sin,
                    "cos": math.cos,
                    "tan": math.tan,
                    "abs": abs,
                    "min": min,
                    "max": max,
                }
                value = float(eval(code, {"__builtins__": {}}, scope))
                return max(value, DENSITY_EPSILON)

            return expression_density

        translated = _translate_density_source(source)
        namespace: dict[str, object] = {
            "__builtins__": {},
            "math": math,
            "float": float,
            "abs": abs,
            "min": min,
            "max": max,
        }
        exec(translated, namespace, namespace)
        compiled = namespace["__compiled_density"]

        def script_density(point: FloatArray) -> float:
            """Evaluate a translated multi-line density script."""

            x = np.asarray(point, dtype=float)
            value = float(compiled(x))
            return max(value, DENSITY_EPSILON)

        return script_density
    except Exception:
        return lambda _point: 1.0
