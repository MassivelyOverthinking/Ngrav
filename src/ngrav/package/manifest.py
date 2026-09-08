#==================================================================================================================
# IMPORTS
#==================================================================================================================

from datetime import datetime, timezone

from pydantic import BaseModel, Field

#==================================================================================================================
# METADATA MANIFESTS
#==================================================================================================================

class BaseInfo(BaseModel):
    """Represents ONNX operator-set dependency."""

    domain: str
    version: int


class OnnxModelInfo(BaseModel):
    """Metadata read directly from the ONNX ModelProto."""

    ir_version: int
    producer_name: str | None = None
    producer_version: str | None = None
    model_version: int | None = None
    graph_name: str | None = None
    opsets: list[BaseInfo] = Field(default_factory=list)


class NgravManifest(BaseModel):
    """Metadata describing a Ngrav package."""

    schema_version: str = "0.1"

    name: str
    original_filename: str
    valid_onnx: bool = False
    onnx: OnnxModelInfo
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))