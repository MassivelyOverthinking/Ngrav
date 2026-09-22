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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

#==================================================================================================================
# NGRAV METADATA: Metadata Base Class
#==================================================================================================================

class NgravMetadataBaseModel(BaseModel):
    """
    Base class for Ngrav medata model_info ==> Inherits from Pydantic BaseModel
    
    Provides model_info serialization functionality to all children manifest models.
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
    # Rich display
    #==================================================================================================================

    def display(self, console: Console | None = None) -> None:
        """
        Display intenral metadata using Rich formatting.
        """

        if console is None:
            console = Console()

        console.print(
            Panel(
                str(self),
                title=self.__class__.__name__,
                border_style="cyan"
            )
        )

    #==================================================================================================================
    # Dunder-methods
    #==================================================================================================================

    def _repr_fields(self) -> str:
        values = self.model_dump(
            mode="python",
            exclude_none=True
        )

        formatted_str = ", ".join(
            f"{key}={value!r}"
            for key, value in values.items()
        )

        return formatted_str

    def __repr__(self) -> str:
        """Return informative developer-facing string representation of manife component"""
        return f"{self.__class__.__name__}({self._repr_fields()})"

    def __str__(self) -> str:
        """Return readable and declarative string representation of manife component"""

        values = self.model_dump(
            mode="python",
            exclude_none=True,
        )

        lines = [self.__class__.__name__]

        for key, value in values.items():
            lines.append(f" {key}: {value}")

        return "\n".join(lines)

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

    runtime_info: runtime_infoInfo
    platform: PlatformInfo
    model_info: OnnxModelInfo
    history_info: ExecutionInfo

    # --------------------------------------------------------------------------------------------------------------
    # Rich display
    # --------------------------------------------------------------------------------------------------------------

    def display(self, console: Console | None = None) -> None:
        """Display the complete manifest metadata using Rich."""

        if console is None:
            console = Console()

        console.print()

        console.print(
            Panel(
                f"[bold cyan]{self.name}[/bold cyan]\n"
                f"{self.description or 'No description'}",
                title="NGRAV MANIFEST",
                border_style="cyan",
            )
        )

        # ------------------------------------------------------------------------------------------
        # General information
        # ------------------------------------------------------------------------------------------

        General_info = Table(
            title="General",
            show_header=False,
            expand=True,
        )

        General_info.add_column("Property", style="cyan")
        General_info.add_column("Value")

        General_info.add_row("Format", self.format)
        General_info.add_row("Manifest version", self.manifest_version)
        General_info.add_row("Original file", self.original_filename)
        General_info.add_row(
            "Valid ONNX",
            "[green]Yes[/green]" if self.valid_onnx else "[red]No[/red]",
        )
        General_info.add_row("Created", self.created_at.isoformat())
        General_info.add_row("Updated", self.updated_at.isoformat())

        console.print(General_info)

        # ------------------------------------------------------------------------------------------
        # Model information
        # ------------------------------------------------------------------------------------------

        model_info = Table(
            title="ONNX Model",
            show_header=False,
            expand=True,
        )

        model_info.add_column("Property", style="cyan")
        model_info.add_column("Value")

        model_info.add_row("IR version", str(self.model_info.ir_version))
        model_info.add_row(
            "Producer",
            self.model_info.producer_name or "Unknown",
        )
        model_info.add_row(
            "Producer version",
            self.model_info.producer_version or "Unknown",
        )
        model_info.add_row(
            "Model version",
            str(self.model_info.model_version)
            if self.model_info.model_version is not None
            else "None",
        )
        model_info.add_row(
            "Graph",
            self.model_info.graph_name or "None",
        )
        model_info.add_row(
            "Opsets",
            str(len(self.model_info.opsets)),
        )

        console.print(model_info)

        # ------------------------------------------------------------------------------------------
        # Runtime Information
        # ------------------------------------------------------------------------------------------

        runtime_info = Table(
            title="runtime_info",
            show_header=False,
            expand=True,
        )

        runtime_info.add_column("Property", style="cyan")
        runtime_info.add_column("Value")

        runtime_info.add_row(
            "Engine",
            self.runtime_info.engine_name,
        )
        runtime_info.add_row(
            "Engine version",
            self.runtime_info.engine_version,
        )
        runtime_info.add_row(
            "Python",
            self.runtime_info.python_version,
        )
        runtime_info.add_row(
            "Implementation",
            self.runtime_info.python_implementation,
        )

        console.print(runtime_info)

        # ------------------------------------------------------------------------------------------
        # Execution History Information
        # ------------------------------------------------------------------------------------------

        history_info = Table(
            title="Execution History",
            show_header=False,
            expand=True,
        )

        history_info.add_column("Property", style="cyan")
        history_info.add_column("Value")

        history_info.add_row(
            "Total executions",
            str(self.history_info.total_count),
        )
        history_info.add_row(
            "Successful",
            f"[green]{self.history_info.successes}[/green]",
        )
        history_info.add_row(
            "Failed",
            f"[red]{self.history_info.failures}[/red]",
        )
        history_info.add_row(
            "Total latency",
            f"{self.history_info.total_latency_ms:.2f} ms",
        )
        history_info.add_row(
            "Average latency",
            f"{self.history_info.average_latency_ms:.2f} ms",
        )

        console.print(history_info)

        if self.history_info.last_execution:
            last = self.history_info.last_execution

            last_execution = Table(
                title="Last Execution",
                show_header=False,
                expand=True,
            )

            last_execution.add_column("Property", style="cyan")
            last_execution.add_column("Value")

            last_execution.add_row("ID", last.id)

            status_style = (
                "green"
                if last.status == "success"
                else "red"
            )

            last_execution.add_row(
                "Status",
                f"[{status_style}]{last.status}[/{status_style}]",
            )

            last_execution.add_row(
                "Duration",
                f"{last.duration_ms:.2f} ms",
            )

            last_execution.add_row(
                "Timestamp",
                last.timestamp.isoformat(),
            )

            console.print(last_execution)

        console.print()

    # --------------------------------------------------------------------------------------------------------------
    # Dunder Methods
    # --------------------------------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"NgravManifest("
            f"name={self.name!r}, "
            f"format={self.format!r}, "
            f"manifest_version={self.manifest_version!r}, "
            f"original_filename={self.original_filename!r}, "
            f"valid_onnx={self.valid_onnx!r}"
            f")"
        )

    def __str__(self) -> str:
        lines = [
            "NgravManifest",
            "────────────────────────────────────────────────────────",
            f"Name:               {self.name}",
            f"Description:        {self.description or 'None'}",
            f"Format:             {self.format}",
            f"Manifest version:   {self.manifest_version}",
            f"Original file:      {self.original_filename}",
            f"Valid ONNX:         {'Yes' if self.valid_onnx else 'No'}",
            "",
            "Model",
            f"  IR version:       {self.model_info.ir_version}",
            f"  Producer:         {self.model_info.producer_name or 'Unknown'}"
            f"{' ' + self.model_info.producer_version if self.model_info.producer_version else ''}",
            f"  Model version:    {self.model_info.model_version or 'None'}",
            f"  Graph:            {self.model_info.graph_name or 'None'}",
            f"  Opsets:           {len(self.model_info.opsets)}",
            "",
            "runtime_info",
            f"  Engine:            {self.runtime_info.engine_name} "
            f"{self.runtime_info.engine_version}",
            f"  Python:            {self.runtime_info.python_version} "
            f"({self.runtime_info.python_implementation})",
            "",
            "Platform",
            f"  OS:                {self.platform.operating_system}",
            f"  Distribution:      {self.platform.distribution} "
            f"{self.platform.distribution_version}",
            f"  Architecture:      {self.platform.architecture}",
            f"  CPU:               {self.platform.cpu_architecture}",
            f"  CPU cores:         {self.platform.cpu_cores}",
            f"  Dependencies:      {len(self.platform.dependencies)}",
            "",
            "Execution History",
            f"  Total executions:  {self.history_info.total_count}",
            f"  Successful:        {self.history_info.successes}",
            f"  Failed:            {self.history_info.failures}",
            f"  Average latency:   {self.history_info.average_latency_ms:.2f} ms",
        ]

        if self.history_info.last_execution:
            last = self.history_info.last_execution

            lines.extend([
                "",
                "Last execution",
                f"  ID:                {last.id}",
                f"  Status:            {last.status}",
                f"  Duration:          {last.duration_ms:.2f} ms",
                f"  Timestamp:         {last.timestamp.isoformat()}",
            ])

        lines.extend([
            "",
            f"Created:            {self.created_at.isoformat()}",
            f"Updated:            {self.updated_at.isoformat()}",
            "────────────────────────────────────────────────────────",
        ])

        return "\n".join(lines)

    

#==================================================================================================================
# NGRAV METADATA: Onnx model_info info
#==================================================================================================================

class OnnxModelInfo(NgravMetadataBaseModel):
    """Metadata read directly from the ONNX ModelProto."""

    ir_version: int
    producer_name: str | None = None
    producer_version: str | None = None
    model_version: int | None = None
    graph_name: str | None = None
    opsets: list[BaseInfo] = Field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"OnnxModelInfo("
            f"ir_version={self.ir_version}, "
            f"producer_name={self.producer_name!r}, "
            f"producer_version={self.producer_version!r}, "
            f"graph_name={self.graph_name!r}, "
            f"opsets={len(self.opsets)}"
            f")"
        )

    def __str__(self) -> str:
        lines = [
            "OnnxModelInfo",
            "────────────────────────────────────────",
            f"IR version:       {self.ir_version}",
            f"Producer:          {self.producer_name or 'Unknown'}",
            f"Producer version:  {self.producer_version or 'Unknown'}",
            f"Model version:     {self.model_version or 'None'}",
            f"Graph:             {self.graph_name or 'None'}",
            f"Opsets:            {len(self.opsets)}",
        ]

        if self.opsets:
            lines.append("")
            lines.append("Operator sets:")

            for opset in self.opsets:
                lines.append(
                    f"  • {opset.domain}: {opset.version}"
                )

        lines.append("────────────────────────────────────────")

        return "\n".join(lines)

class BaseInfo(NgravMetadataBaseModel):
    """Represents ONNX operator-set dependency."""

    domain: str
    version: int

    def __repr__(self) -> str:
        return (
            f"BaseInfo("
            f"domain={self.domain!r}, "
            f"version={self.version}"
            f")"
        )

    def __str__(self) -> str:
        return (
            f"Operator Set\n"
            f"  Domain:   {self.domain}\n"
            f"  Version:  {self.version}"
        )

#==================================================================================================================
# NGRAV METADATA: Execution info
#==================================================================================================================

class ExecutionInfo(NgravMetadataBaseModel):
    """Metadata representing core elements of NgravPacket internal execution history_info"""

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

    def __repr__(self) -> str:
        return (
            f"ExecutionInfo("
            f"total_count={self.total_count}, "
            f"successes={self.successes}, "
            f"failures={self.failures}, "
            f"average_latency_ms={self.average_latency_ms:.2f}"
            f")"
        )

    def __str__(self) -> str:
        lines = [
            "ExecutionInfo",
            "────────────────────────────────────────",
            f"Total executions:  {self.total_count}",
            f"Successful:        {self.successes}",
            f"Failed:            {self.failures}",
            f"Total latency:     {self.total_latency_ms:.2f} ms",
            f"Average latency:   {self.average_latency_ms:.2f} ms",
        ]

        if self.last_execution:
            lines.extend([
                "",
                "Last execution:",
                f"  ID:          {self.last_execution.id}",
                f"  Status:      {self.last_execution.status}",
                f"  Duration:    {self.last_execution.duration_ms:.2f} ms",
                f"  Timestamp:   {self.last_execution.timestamp.isoformat()}",
            ])

        lines.append("────────────────────────────────────────")

        return "\n".join(lines)

class LastExecutionEntryInfo(NgravMetadataBaseModel):
    id: str
    timestamp: datetime
    status: str
    duration_ms: float

    def __repr__(self) -> str:
        return (
            f"LastExecutionEntryInfo("
            f"id={self.id!r}, "
            f"status={self.status!r}, "
            f"duration_ms={self.duration_ms:.2f}"
            f")"
        )

    def __str__(self) -> str:
        return (
            "LastExecutionEntryInfo\n"
            "────────────────────────────────────────\n"
            f"ID:          {self.id}\n"
            f"Status:      {self.status}\n"
            f"Duration:    {self.duration_ms:.2f} ms\n"
            f"Timestamp:   {self.timestamp.isoformat()}\n"
            "────────────────────────────────────────"
        )

#==================================================================================================================
# NGRAV METADATA: runtime_info & Platform info
#==================================================================================================================

class runtime_infoInfo(NgravMetadataBaseModel):
    """Metadata representing the runtime_info environment and model_info execution"""

    engine_name: str = "onnxruntime_info"
    engine_version: str

    python_version: str
    python_implementation: str

    def __repr__(self) -> str:
        return (
            f"runtime_infoInfo("
            f"engine_name={self.engine_name!r}, "
            f"engine_version={self.engine_version!r}, "
            f"python_version={self.python_version!r}, "
            f"python_implementation={self.python_implementation!r}"
            f")"
        )

    def __str__(self) -> str:
        return (
            "runtime_infoInfo\n"
            "────────────────────────────────────────\n"
            f"Engine:           {self.engine_name}\n"
            f"Engine version:   {self.engine_version}\n"
            f"Python:           {self.python_version}\n"
            f"Implementation:   {self.python_implementation}\n"
            "────────────────────────────────────────"
        )

class DependencyInfo(NgravMetadataBaseModel):
    """Metadata representing current version dependecies"""

    name: str
    version: str

    def __repr__(self) -> str:
        return (
            f"DependencyInfo("
            f"name={self.name!r}, "
            f"version={self.version!r}"
            f")"
        )

    def __str__(self) -> str:
        return f"{self.name} {self.version}"

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

    def __repr__(self) -> str:
        return (
            f"PlatformInfo("
            f"operating_system={self.operating_system!r}, "
            f"distribution={self.distribution!r}, "
            f"distribution_version={self.distribution_version!r}, "
            f"architecture={self.architecture!r}, "
            f"cpu_architecture={self.cpu_architecture!r}, "
            f"cpu_cores={self.cpu_cores}, "
            f"dependencies={len(self.dependencies)}"
            f")"
        )

    def __str__(self) -> str:
        return (
            "PlatformInfo\n"
            "────────────────────────────────────────\n"
            f"Operating system:  {self.operating_system}\n"
            f"Distribution:      {self.distribution}\n"
            f"Version:            {self.distribution_version}\n"
            f"Architecture:      {self.architecture}\n"
            f"CPU architecture:  {self.cpu_architecture}\n"
            f"CPU cores:         {self.cpu_cores}\n"
            f"Dependencies:      {len(self.dependencies)}\n"
            "────────────────────────────────────────"
        )
        
