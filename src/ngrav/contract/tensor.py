#==================================================================================================================
# IMPORTS
#==================================================================================================================

from __future__ import annotations

import numpy as np

from enum import StrEnum

#==================================================================================================================
# TENSOR MAPPING: NumPy <---> StrEnum
#==================================================================================================================

class TensorDType(StrEnum):
    """Mapping NumPy data-types to simple Str representation"""

    # Boolean Datatypes
    BOOL = "bool"

    # Integer Datatypes
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"

    # Unsigned Integer Datatypes
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"

    # Floating-point Integer Datatypes
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"

    # Complex Integer Datatypes
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"

NUMPY_DTYPE_MAP: dict[TensorDType, np.dtype] = {
    TensorDType.BOOL: np.dtype(np.bool_),

    TensorDType.INT8: np.dtype(np.int8),
    TensorDType.INT16: np.dtype(np.int16),
    TensorDType.INT32: np.dtype(np.int32),
    TensorDType.INT64: np.dtype(np.int64),

    TensorDType.UINT8: np.dtype(np.uint8),
    TensorDType.UINT16: np.dtype(np.uint16),
    TensorDType.UINT32: np.dtype(np.uint32),
    TensorDType.UINT64: np.dtype(np.uint64),

    TensorDType.FLOAT16: np.dtype(np.float16),
    TensorDType.FLOAT32: np.dtype(np.float32),
    TensorDType.FLOAT64: np.dtype(np.float64),

    TensorDType.COMPLEX64: np.dtype(np.complex64),
    TensorDType.COMPLEX128: np.dtype(np.complex128),
}

#==================================================================================================================
# TENSOR MAPPING: Validation method
#==================================================================================================================

def validate_numpy_dtype(dtype: TensorDType) -> np.dtype:
    return NUMPY_DTYPE_MAP[dtype]