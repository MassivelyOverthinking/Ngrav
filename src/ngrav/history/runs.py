#==================================================================================================================
# IMPORTS
#==================================================================================================================

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

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

# Pydantic Schema ==> ONNX runtime execution data
class ExecutionRuntimeInfo(BaseModel):
    """
    Metadata schema ONNX runtime data.
    """

    model_config = ConfigDict(frozen=True)

    runtime_version: str
    providers: list[str]
    python_version: str
    platform: str

# Pydantic Schema ==> Runtime-related error/exception data
class ExecutionErrorInfo(BaseModel):
    """
    Metadata schema for execution-related error/exception data.
    """

    model_config = ConfigDict(frozen=True)

    type: str
    message: str
    code: str | None = None

# Pydantic Schema ==> Final execution snapshot
class ExecutionRecord(BaseModel):
    """
    Metadata schema for complete execution-related data.
    """
    
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

# Data Class ==> Complete execution history
class ExecutionHistory(BaseModel):
    """
    Data class depicting the complete execution history of NgravPacket.
    """

    records: list[ExecutionRecord] = Field(default_factory=list)

    @classmethod
    def create(cls) -> ExecutionHistory:
        return cls()

    def append(self, record: ExecutionRecord) -> None:
        if not isinstance(record, ExecutionRecord):
            raise TypeError(f"Records must be of Type: ExecutionRecord - REceived dtype: {type(record).__name__}")

        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, index: int) -> ExecutionRecord:
        return self.records[index]

