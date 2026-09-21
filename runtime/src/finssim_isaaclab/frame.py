from __future__ import annotations

from typing import Sequence

import numpy as np


# FinsROV: x forward, y up, z left.
# Isaac WarpAUV: x forward, y left, z up.
FINS_TO_ISAAC_BASIS = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def _quat_xyzw_to_matrix(quat_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = np.linalg.norm([x, y, z, w])
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = np.asarray([x, y, z, w], dtype=np.float64) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.asarray([x, y, z, w], dtype=np.float64)
    return (q / max(np.linalg.norm(q), 1e-12)).astype(np.float32)


def convert_quaternion_xyzw_to_wxyz(quat_xyzw: Sequence[float], basis: np.ndarray = FINS_TO_ISAAC_BASIS) -> np.ndarray:
    rotation = np.asarray(basis, dtype=np.float64) @ _quat_xyzw_to_matrix(quat_xyzw) @ np.asarray(basis, dtype=np.float64).T
    x, y, z, w = _matrix_to_quat_xyzw(rotation)
    return np.asarray([w, x, y, z], dtype=np.float32)


def transform_body_angular_vector(vector_fins: Sequence[float]) -> np.ndarray:
    """Transform FinsROV angular velocity into Isaac WarpAUV coordinates.

    The FinsROV-to-Isaac map exchanges up and left, so its determinant is
    -1. Angular velocity is an axial vector and therefore requires the
    determinant factor in addition to the ordinary vector basis transform.
    """

    basis = FINS_TO_ISAAC_BASIS
    return (-basis @ np.asarray(vector_fins, dtype=np.float64).reshape(3)).astype(np.float32)
