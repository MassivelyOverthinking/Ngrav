#==================================================================================================================
# IMPORTS
#==================================================================================================================

from __future__ import annotations

import json
import yaml
import platform
import sys

from typing import Any
from pathlib import Path
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field
import onnxruntime as ort

from ngrav.utility import PathInput

#==================================================================================================================
# EXECUTION HISTORY: Schemas & Pydantic Models
#==================================================================================================================

# Enum-Type ==> NgravPacket execution status
class ExecutionStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"

# Pydantic Schema ==> Tensor Output and Input
class TensorExecutionInfo(BaseModel):
    """
    Metadata schema for model input and output data.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    shape: list[int] | None = None
    dtype: str
    nbytes: int | None = None

class ExecutionRuntimeInfo(BaseModel):
    """
    Metadata schema ONNX runtime data.
    """

    model_config = ConfigDict(frozen=True)

    runtime_version: str
    provider: list[str]
    python_version: str
    platform: str

class ExecutionErrorInfo(BaseModel):
    """
    Metadata schme for execution-related error/exception data.
    """

    model_config = ConfigDict(frozen=True)

    type: str
    message: str
    code: str | None = None

class ExecutionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    execution_id: UUID = Field(default_factory=uuid4)

    status: ExecutionStatus

    started_at: datetime
    finished_at: datetime
    duration_ms: float = Field(ge=0)

    fingerprint: str | None = None
    runtime: ExecutionRuntimeInfo

    inputs: list[TensorExecutionInfo] = Field(default_factory=list)
    outputs: list[TensorExecutionInfo] = Field(default_factory=list)
    input_bytes: int = Field(decimal_places=0, ge=0)
    output_bytes: int = Field(decimal_places=0, ge=0)

    error: ExecutionErrorInfo | None = None

