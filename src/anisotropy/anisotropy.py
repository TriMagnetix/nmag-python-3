"""
This file contains the definition of the PredefinedAnisotropy class
and some useful functions for defining the magnetic anisotropy
of a magnetic material.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

EnergyFunction = Callable[[ArrayLike], float]
AnisotropyStringifier = Callable[["PredefinedAnisotropy"], str]


def _normalize(v: ArrayLike) -> NDArray[np.float64]:
    """Helper function to normalize a NumPy vector, handling the zero vector case."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return v / norm


def want_anisotropy(x: object, want_function: bool = True) -> None:
    """
    Checks if x is a valid PredefinedAnisotropy object.

    Args:
        x: The object to check.
        want_function (bool): If True, requires the object to have an
                              associated energy function.
    """
    if not isinstance(x, PredefinedAnisotropy):
        # Use TypeError for incorrect argument types
        raise TypeError(f"Operand must be a PredefinedAnisotropy object, not {type(x).__name__}")
    if want_function and not x.has_function():
        # Use ValueError for correct type but invalid state
        raise ValueError("Cannot operate on an anisotropy object that lacks an energy function.")


class PredefinedAnisotropy:
    """
    Contains an anisotropy energy function and its approximation order.
    """

    def __init__(
        self,
        function: EnergyFunction | None = None,
        order: int | None = None,
        anis_type: str = "functional",
        axis1: ArrayLike | None = None,
        axis2: ArrayLike | None = None,
        axis3: ArrayLike | None = None,
        K1: float | None = None,
        K2: float | None = None,
        K3: float | None = None,
        stringifier: AnisotropyStringifier | None = None,
    ) -> None:
        if function is None and order is None:
            # Use ValueError for invalid argument values
            raise ValueError("PredefinedAnisotropy requires either a 'function' or an 'order'.")
        self.anis_type = anis_type
        self.function = function
        self.order = order
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        # Ensure axes are NumPy arrays if they exist
        self.axis1 = np.asarray(axis1) if axis1 is not None else None
        self.axis2 = np.asarray(axis2) if axis2 is not None else None
        self.axis3 = np.asarray(axis3) if axis3 is not None else None
        self._str_extra = stringifier

    def has_function(self) -> bool:
        """Returns True if the anisotropy has an energy function."""
        return self.function is not None

    def __str__(self) -> str:
        s = self.anis_type
        if self._str_extra:
            s += f", {self._str_extra(self)}"
        return f"<PredefinedAnisotropy:{s}>"

    def __repr__(self) -> str:
        s = self._str_extra(self) if self._str_extra else "?"
        return f'PredefinedAnisotropy(anis_type="{self.anis_type}", {s})'

    def __neg__(self) -> PredefinedAnisotropy:
        """Unary operator -"""
        want_anisotropy(self)
        function = self.function
        assert function is not None

        def neg_function(m: ArrayLike) -> float:
            return -float(function(m))

        return PredefinedAnisotropy(neg_function, self.order)

    def __pos__(self) -> PredefinedAnisotropy:
        """Unary operator +"""
        return self

    def __add__(self, y: object) -> PredefinedAnisotropy:
        """Addition operator"""
        want_anisotropy(self)
        want_anisotropy(y)
        assert isinstance(y, PredefinedAnisotropy)
        left_function = self.function
        right_function = y.function
        left_order = self.order
        right_order = y.order
        assert left_function is not None and right_function is not None
        assert left_order is not None and right_order is not None

        def f(m: ArrayLike) -> float:
            return float(left_function(m) + right_function(m))

        o = max(left_order, right_order)
        return PredefinedAnisotropy(f, o)

    def __sub__(self, y: object) -> PredefinedAnisotropy:
        """Subtraction operator"""
        want_anisotropy(self)
        want_anisotropy(y)
        assert isinstance(y, PredefinedAnisotropy)
        left_function = self.function
        right_function = y.function
        left_order = self.order
        right_order = y.order
        assert left_function is not None and right_function is not None
        assert left_order is not None and right_order is not None

        def f(m: ArrayLike) -> float:
            return float(left_function(m) - right_function(m))

        o = max(left_order, right_order)
        return PredefinedAnisotropy(f, o)


def uniaxial_anisotropy(
    axis: ArrayLike,
    K1: float,
    K2: float = 0.0,
) -> PredefinedAnisotropy:
    """
    Returns a predefined anisotropy for a uniaxial energy density term:
        E_anis = - K1 * <axis, m>^2 - K2 * <axis, m>^4
    where 'm' is the normalized magnetization.

    Args:
        axis (list or np.ndarray): Easy or hard axis (will be normalized).
        K1 (float): Second-order anisotropy constant.
        K2 (float): Fourth-order anisotropy constant (default is 0).
    """
    # Normalize the easy axis using NumPy
    axis = _normalize(axis)

    # Build the anisotropy function using NumPy's dot product
    def f(m: ArrayLike) -> float:
        m = np.asarray(m, dtype=float)
        a = np.dot(axis, m)
        return float(-K1 * a**2 - K2 * a**4)

    order = 4 if K2 else 2

    def stringifier(a: PredefinedAnisotropy) -> str:
        assert a.axis1 is not None
        s = f"axis={list(a.axis1)}, K1={a.K1}"
        if a.K2 != 0.0:
            s += f", K2={a.K2}"
        return s

    return PredefinedAnisotropy(
        anis_type="uniaxial",
        function=f,
        order=order,
        axis1=axis,
        K1=K1,
        K2=K2,
        stringifier=stringifier,
    )


def cubic_anisotropy(
    axis1: ArrayLike,
    axis2: ArrayLike,
    K1: float,
    K2: float = 0.0,
    K3: float = 0.0,
) -> PredefinedAnisotropy:
    """
    Returns a predefined anisotropy for a cubic energy density term:
        E_anis = K1 * (<axis1,m>^2 <axis2,m>^2 + <axis1,m>^2 <axis3,m>^2 + <axis2,m>^2 <axis3,m>^2)
           + K2 * (<axis1,m>^2 <axis2,m>^2 <axis3,m>^2)
           + K3 * (<axis1,m>^4 <axis2,m>^4 + <axis1,m>^4 <axis3,m>^4 + <axis2,m>^4 <axis3,m>^4)
    (where `m` is the (normalised) magnetisation.)

    Args:
        axis1 (list or np.ndarray): First cubic axis (will be normalized).
        axis2 (list or np.ndarray): Second cubic axis (will be orthonormalized).
        K1 (float): Fourth-order anisotropy constant.
        K2 (float): Sixth-order anisotropy constant (default is 0).
        K3 (float): Eighth-order anisotropy constant (default is 0).
    """
    # Convert inputs to NumPy arrays
    a1 = np.asarray(axis1, dtype=float)
    a2 = np.asarray(axis2, dtype=float)

    # Build an orthonormal system using NumPy
    a3 = np.cross(a1, a2)
    a2 = np.cross(a3, a1)  # Ensure a2 is orthogonal to a1 and a3

    # Normalize all axes
    axis1 = _normalize(a1)
    axis2 = _normalize(a2)
    axis3 = _normalize(a3)

    # Build anisotropy function using NumPy
    def f(m: ArrayLike) -> float:
        m = np.asarray(m, dtype=float)
        a1_dot_m = np.dot(axis1, m)
        a2_dot_m = np.dot(axis2, m)
        a3_dot_m = np.dot(axis3, m)

        term1 = (a1_dot_m * a2_dot_m) ** 2 + (a1_dot_m * a3_dot_m) ** 2 + (a2_dot_m * a3_dot_m) ** 2
        term2 = (a1_dot_m * a2_dot_m * a3_dot_m) ** 2
        term3 = (a1_dot_m * a2_dot_m) ** 4 + (a1_dot_m * a3_dot_m) ** 4 + (a2_dot_m * a3_dot_m) ** 4

        return float(K1 * term1 + K2 * term2 + K3 * term3)

    def stringifier(a: PredefinedAnisotropy) -> str:
        assert a.axis1 is not None and a.axis2 is not None
        s = f"axis1={list(a.axis1)}, axis2={list(a.axis2)}, K1={a.K1}"
        if a.K2 != 0.0:
            s += f", K2={a.K2}"
        if a.K3 != 0.0:
            s += f", K3={a.K3}"
        return s

    if K3:
        order = 8
    elif K2:
        order = 6
    else:
        order = 4

    return PredefinedAnisotropy(
        anis_type="cubic",
        function=f,
        order=order,
        axis1=axis1,
        axis2=axis2,
        axis3=axis3,
        K1=K1,
        K2=K2,
        K3=K3,
        stringifier=stringifier,
    )
