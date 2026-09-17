#==================================================================================================================
# IMPORTS
#==================================================================================================================

from __future__ import annotations

import platform
import os

from importlib.metadata import distributions
from datetime import datetime, UTC

from pydantic import BaseModel, Field

#==================================================================================================================
# NGRAV METADATA: Core Manifest
#==================================================================================================================

class NgravManifest(BaseModel):
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

#==================================================================================================================
# NGRAV METADATA: Onnx model info
#==================================================================================================================

class OnnxModelInfo(BaseModel):
    """Metadata read directly from the ONNX ModelProto."""

    ir_version: int
    producer_name: str | None = None
    producer_version: str | None = None
    model_version: int | None = None
    graph_name: str | None = None
    opsets: list[BaseInfo] = Field(default_factory=list)

class BaseInfo(BaseModel):
    """Represents ONNX operator-set dependency."""

    domain: str
    version: int

#==================================================================================================================
# NGRAV METADATA: Runtime & Platform info
#==================================================================================================================

class RuntimeInfo(BaseModel):
    """Metadata representing the runtime environment and model execution"""

    engine_name: str = "onnxruntime"
    enngine_version: str

    python_version: str
    python_implementation: str

class DependencyInfo(BaseModel):
    """Metadata representing current version dependecies"""

    name: str
    version: str

class PlatformInfo(BaseModel):
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
        
