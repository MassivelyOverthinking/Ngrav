#==================================================================================================================
# IMPORTS
#==================================================================================================================

from __future__ import annotations

import platform as plt
import numpy as np
import onnxruntime as ort

from typing import Any

from .runs import TensorExecutionInfo, ExecutionRuntimeInfo, ExecutionErrorInfo

#==================================================================================================================
# EXECUTION HISTORY: Builder methods
#==================================================================================================================

def describe_tensor_values(name: str, value: Any) -> TensorExecutionInfo:
    """
    Produce safe metadata regarding input and output Tensor values for Ngrav Execution History.

    Raw tensor-values are deliberately not retained, only a data version.
    """

    if isinstance(value, np.ndarray):
        return TensorExecutionInfo(
            name=name,
            shape=list(value.shape),
            dtype=str(value.dtype),
            nbytes=int(value.nbytes)
        )

    return TensorExecutionInfo(
        name=name,
        dtype=type(value).__name__
    )

def describe_runtime_info(providers: list[str] | None = None) -> ExecutionRuntimeInfo:
    """
    Produce metadata regarding ONNX Runtime inference for Ngrav Execution History.
    """

    return ExecutionRuntimeInfo(
        runtime_version=ort.__version__,
        providers=providers or [],
        python_version=plt.python_version(),
        platform=plt.platform()
    )

def describe_execution_error(exec: Exception) -> ExecutionErrorInfo:
    """
    Produce metadata regarding potential errors/execeptions thrown during inferece session for Ngrav Execution History.
    
    Raw Exceptions are not stored, only a data version.
    """

    error_code = getattr(exec, "code", None)

    return ExecutionErrorInfo(
        type=type(exec).__name__,
        message=str(exec),
        code=error_code,
    )