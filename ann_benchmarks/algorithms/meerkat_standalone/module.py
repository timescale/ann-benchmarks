"""
Meerkat standalone ann-benchmarks module.

Uses the meerkat shared library (libmkt.so) directly via ctypes
for in-memory index build and query without PostgreSQL overhead.
"""

import ctypes
import os
import sys

import numpy as np

from ..base.module import BaseANN

# Search paths for the shared library
_LIB_PATHS = [
    "/usr/local/lib/libmkt.so",
    os.path.join(os.path.dirname(__file__), "libmkt.so"),
]


def _load_lib():
    # Use RTLD_LAZY (1) to avoid eager ifunc resolution crashes
    for path in _LIB_PATHS:
        if os.path.exists(path):
            return ctypes.CDLL(path, mode=1)
    raise RuntimeError(
        "Cannot find libmkt.so. Tried: " + ", ".join(_LIB_PATHS)
    )


class MeerkatStandalone(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._nlist = method_param.get("nlist", 0)
        self._fan_out = method_param.get("fan_out", 0)
        self._centroid_fmt = method_param.get("centroid_fmt", "rabitq")
        self._encode_rabitq = method_param.get("encode_rabitq", 1)
        self._distance_mode = method_param.get("distance_mode", "asymmetric")
        self._scan_mode = method_param.get("scan_mode", "fastscan")
        self._soar_lambda = method_param.get("soar_lambda", 0.0)
        self._boundary_epsilon = method_param.get("boundary_epsilon", 0.0)
        self._nprobe = 10
        self._handle = None

        self._lib = _load_lib()

        self._lib.mkt_handle_create_from_array.restype = ctypes.c_void_p
        self._lib.mkt_handle_create_from_array.argtypes = [
            ctypes.c_void_p,   # vectors
            ctypes.c_uint32,   # nvecs
            ctypes.c_uint32,   # dim
            ctypes.c_uint32,   # nlist
            ctypes.c_uint32,   # fan_out
            ctypes.c_char_p,   # metric
            ctypes.c_char_p,   # centroid_fmt
            ctypes.c_char_p,   # posting_fmt
            ctypes.c_uint32,   # km_nredo
            ctypes.c_uint32,   # km_max_iter
            ctypes.c_double,   # soar_lambda
            ctypes.c_double,   # boundary_epsilon
            ctypes.c_void_p,   # info (nullable)
        ]

        self._lib.mkt_handle_query.restype = ctypes.c_uint32
        self._lib.mkt_handle_query.argtypes = [
            ctypes.c_void_p,   # handle
            ctypes.c_void_p,   # query
            ctypes.c_uint32,   # k
            ctypes.c_uint32,   # nprobe
            ctypes.c_char_p,   # distance_mode
            ctypes.c_int,      # rerank (bool, passed as int for ABI compat)
            ctypes.c_void_p,   # result_ids
        ]

        self._lib.mkt_handle_destroy.restype = None
        self._lib.mkt_handle_destroy.argtypes = [ctypes.c_void_p]

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        nvecs, dim = X.shape

        metric_str = self._metric.encode("utf-8")
        fmt_str = self._centroid_fmt.encode("utf-8")

        print(
            f"Building meerkat index: {nvecs} x {dim}, "
            f"nlist={self._nlist}, metric={self._metric}, "
            f"centroid_fmt={self._centroid_fmt}"
        )
        sys.stdout.flush()

        posting_fmt_str = self._scan_mode.encode("utf-8")

        self._handle = self._lib.mkt_handle_create_from_array(
            X.ctypes.data,
            nvecs,
            dim,
            self._nlist,
            self._fan_out,
            metric_str,
            fmt_str,
            posting_fmt_str,
            0,
            0,
            ctypes.c_double(self._soar_lambda),
            ctypes.c_double(self._boundary_epsilon),
            None,
        )

        if not self._handle:
            raise RuntimeError("mkt_handle_create_from_array failed")

        self._dim = dim
        print("Index built successfully")
        sys.stdout.flush()

    def set_query_arguments(self, nprobe):
        if isinstance(nprobe, (list, tuple)):
            self._nprobe = nprobe[0]
        else:
            self._nprobe = nprobe

    def query(self, q, n):
        v = np.ascontiguousarray(q, dtype=np.float32)
        result_ids = np.empty(n, dtype=np.uint32)

        mode_str = self._distance_mode.encode("utf-8")

        count = self._lib.mkt_handle_query(
            self._handle,
            v.ctypes.data,
            n,
            self._nprobe,
            mode_str,
            True,
            result_ids.ctypes.data,
        )

        return result_ids[:count].tolist()

    def done(self):
        if self._handle:
            self._lib.mkt_handle_destroy(self._handle)
            self._handle = None

    def __str__(self):
        parts = [
            f"metric={self._metric}",
            f"nlist={self._nlist}",
            f"fmt={self._centroid_fmt}",
            f"mode={self._distance_mode}",
            f"nprobe={self._nprobe}",
        ]
        if self._soar_lambda > 0:
            parts.append(f"soar={self._soar_lambda}")
        if self._boundary_epsilon > 0:
            parts.append(f"boundary={self._boundary_epsilon}")
        return f"MeerkatStandalone({', '.join(parts)})"
