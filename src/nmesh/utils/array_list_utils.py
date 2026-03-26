import numpy as np

def array_filter(p, arr):
    arr = np.asanyarray(arr)
    # Apply predicate to create a boolean mask.
    mask = np.vectorize(p)(arr)
    return arr[mask]

def array_position(x, arr, start=0):
    arr = np.asanyarray(arr)
    sub_arr = arr[start:]
    indices = np.where(sub_arr == x)[0]
    if len(indices) > 0:
        return int(indices[0] + start)
    return -1

def array_position_if(p, arr, start=0):
    arr = np.asanyarray(arr)
    sub_arr = arr[start:]
    mask = np.vectorize(p)(sub_arr)
    indices = np.where(mask)[0]
    if len(indices) > 0:
        return int(indices[0] + start)
    return -1

def array_one_shorter(arr, pos):
    return np.delete(np.asanyarray(arr), pos)

def determinant(mx):
    return float(np.linalg.det(np.asanyarray(mx)))

def inverse(mx):
    return np.linalg.inv(np.asanyarray(mx))

def det_and_inv(mx):
    arr = np.asanyarray(mx)
    det = np.linalg.det(arr)
    inv = np.linalg.inv(arr)
    return float(det), inv

def cross_product_3d(v1, v2):
    return np.cross(np.asanyarray(v1), np.asanyarray(v2))
