#==================================================================================================================
# IMPORTS
#==================================================================================================================

from __future__ import annotations

import json
import yaml
import platform
import os

from importlib.metadata import distributions
from datetime import datetime, UTC
from typing import Any

from pydantic import BaseModel, Field

#==================================================================================================================
# NGRAV METADATA: Metadata Base Class
#==================================================================================================================

class NgravMetadataBaseModel(BaseModel):
    """
    Base class for Ngrav medata model ==> Inherits from Pydantic BaseModel
    
    Provides model serialization functionality to all children manifest models.
    """

    #==================================================================================================================
    # Serialization
    #==================================================================================================================

    def to_dict(self) -> dict[str, Any]:
        """Return metadata as standard Python dictionary"""

        return self.model_dump(mode="json")

    def to_json(self, *, indent: int = 2) -> str:
        """Return metadata as standard JSON object"""
            
        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False
        )

    def to_yaml(self) -> str:
        """Return metadata as standard YAML document"""
                
        return yaml.safe_dump(
            self.to_dict(),
            sort_keys=False,
            allow_unicode=True
        )

#==================================================================================================================
# NGRAV METADATA: Core Manifest
#==================================================================================================================

class NgravManifest(NgravMetadataBaseModel):
    """Metadata describing a Ngrav package."""

    manifest_version: str = "0.1"
    format: str = "ngrav"

    name: str
    description: str | None = None
    original_filename: str
    valid_onnx: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    runtime: RuntimeInfo
    platform: PlatformInfo
    model: OnnxModelInfo
    history: ExecutionInfo

#==================================================================================================================
# NGRAV METADATA: Onnx model info
#==================================================================================================================

class OnnxModelInfo(NgravMetadataBaseModel):
    """Metadata read directly from the ONNX ModelProto."""

    ir_version: int
    producer_name: str | None = None
    producer_version: str | None = None
    model_version: int | None = None
    graph_name: str | None = None
    opsets: list[BaseInfo] = Field(default_factory=list)

class BaseInfo(NgravMetadataBaseModel):
    """Represents ONNX operator-set dependency."""

    domain: str
    version: int

#==================================================================================================================
# NGRAV METADATA: Execution info
#==================================================================================================================

class ExecutionInfo(NgravMetadataBaseModel):
    """Metadata representing core elements of NgravPacket internal execution history"""

    total_count: int = 0
    successes: int = 0
    failures: int = 0

    total_latency_ms: float = 0.0
    average_latency_ms: float = 0.0

    last_execution: LastExecutionEntryInfo | None = None

    def add_count(self, is_success: bool = True) -> None:
        if is_success:
            self.successes += 1
        else:
            self.failures += 1

        self.total_count += 1

    def add_latency(self, duration: float) -> None:
        self.total_latency_ms += duration

        if self.total_count > 0:
            self.average_latency_ms = (self.average_latency_ms / self.total_count)

    def update_last_entry(self, id: str, duration: float, is_success: bool = True) -> None:
        self.last_execution = LastExecutionEntryInfo(
            id=id,
            timestamp=datetime.now(UTC),
            status="success" if is_success else "failure",
            duration_ms=duration
        )

class LastExecutionEntryInfo(NgravMetadataBaseModel):
    id: str
    timestamp: datetime
    status: str
    duration_ms: float

#==================================================================================================================
# NGRAV METADATA: Runtime & Platform info
#==================================================================================================================

class RuntimeInfo(NgravMetadataBaseModel):
    """Metadata representing the runtime environment and model execution"""

    engine_name: str = "onnxruntime"
    engine_version: str

    python_version: str
    python_implementation: str

class DependencyInfo(NgravMetadataBaseModel):
    """Metadata representing current version dependecies"""

    name: str
    version: str

class PlatformInfo(NgravMetadataBaseModel):
    """Metadata representing the current platform and dependencies"""

    operating_system: str 
    distribution: str
    distribution_version: str
    architecture: str

    cpu_architecture: str
    cpu_cores: int

    dependencies: list[DependencyInfo] = Field(default_factory=list)

    @staticmethod
    def get_data() -> PlatformInfo:
        system = platform.system().lower()

        distribution = ""
        distribution_version = ""

        if system == "linux":
            try:
                os_release = platform.freedesktop_os_release()
                distribution = os_release.get("NAME", "")
                distribution_version = os_release.get("VERSION_ID", "")
            except OSError:
                # Fallback for systems where /etc/os-release is unavailable
                distribution = platform.system()
                distribution_version = platform.release()

        elif system == "windows":
            distribution = "Windows"
            distribution_version = platform.release()

        elif system == "darwin":
            distribution = "macOS"
            distribution_version = platform.mac_ver()[0]

        dependencies_info = [
            DependencyInfo(
                name=dist.metadat["name"],
                version=dist.version,
            )
            for dist in distributions()
        ]

        return PlatformInfo(
            operating_system=system,
            distribution=distribution,
            distribution_version=distribution_version,
            architecture=platform.machine(),
            cpu_architecture=platform.processor() or platform.machine(),
            cpu_cores=os.cpu_count() or 1,
            dependencies=dependencies_info
        )
        
