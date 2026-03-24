import numpy as np
from typing import Callable, Sequence, TypeVar
from numpy.typing import NDArray

T = TypeVar('T')

def array_filter(p: Callable[[T], bool], arr: Sequence[T]) -> NDArray:
    """Filter array elements using a predicate function.

    Args:
        p: Predicate function returning True for elements to keep
        arr: Input array or sequence

    Returns:
        NumPy array containing only elements where p(x) is True
    """
    arr_np = np.asanyarray(arr)
    mask = np.array([p(x) for x in arr_np], dtype=bool)
    return arr_np[mask]

def array_position(x: T, arr: Sequence[T], start: int = 0) -> int:
    """Find first position of element x in array starting from index start.

    Args:
        x: Element to find
        arr: Array to search
        start: Starting index (default 0)

    Returns:
        Index of first occurrence, or -1 if not found
    """
    arr_np = np.asanyarray(arr)
    sub_arr = arr_np[start:]
    indices = np.where(sub_arr == x)[0]
    if len(indices) > 0:
        return int(indices[0] + start)
    return -1

def array_position_if(p: Callable[[T], bool], arr: Sequence[T], start: int = 0) -> int:
    """Find first position where predicate p is True, starting from index start.

    Args:
        p: Predicate function
        arr: Array to search
        start: Starting index (default 0)

    Returns:
        Index of first match, or -1 if not found
    """
    arr_np = np.asanyarray(arr)
    sub_arr = arr_np[start:]
    mask = np.array([p(x) for x in sub_arr], dtype=bool)
    indices = np.where(mask)[0]
    if len(indices) > 0:
        return int(indices[0] + start)
    return -1

def array_one_shorter(arr: Sequence[T], pos: int) -> NDArray:
    """Remove element at position pos from array.

    Args:
        arr: Input array
        pos: Index of element to remove

    Returns:
        New array with element removed
    """
    return np.delete(np.asanyarray(arr), pos)

def determinant(mx: Sequence[Sequence[float]]) -> float:
    """Compute determinant of a matrix.

    Args:
        mx: Square matrix

    Returns:
        Determinant as float
    """
    return float(np.linalg.det(np.asanyarray(mx)))

def inverse(mx: Sequence[Sequence[float]]) -> NDArray:
    """Compute inverse of a matrix.

    Args:
        mx: Square invertible matrix

    Returns:
        Inverse matrix as NumPy array
    """
    return np.linalg.inv(np.asanyarray(mx))

def det_and_inv(mx: Sequence[Sequence[float]]) -> tuple[float, NDArray]:
    """Compute determinant and inverse of a matrix simultaneously.

    Args:
        mx: Square invertible matrix

    Returns:
        Tuple of (determinant, inverse_matrix)
    """
    arr = np.asanyarray(mx)
    det = np.linalg.det(arr)
    inv = np.linalg.inv(arr)
    return float(det), inv

def cross_product_3d(v1: Sequence[float], v2: Sequence[float]) -> NDArray:
    """Compute 3D cross product of two vectors.

    Args:
        v1: First 3D vector
        v2: Second 3D vector

    Returns:
        Cross product v1 × v2 as NumPy array
    """
    return np.cross(np.asanyarray(v1), np.asanyarray(v2))
