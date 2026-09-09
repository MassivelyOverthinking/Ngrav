#==================================================================================================================
# IMPORTS
#==================================================================================================================

from __future__ import annotations

import shutil
from pathlib import Path

import onnx
import yaml
from onnx import ModelProto, TensorProto

from ..utility import (
    PathInput
)

from ..exceptions import (
    InvalidOnnxPackageError,
    InvalidOnnxModelError,
    UnsupportedOnnxModelError
)

from manifest import (
    NgravManifest,
    BaseInfo,
    OnnxModelInfo
)

#==================================================================================================================
# NGRAV BASE CONTAINER
#==================================================================================================================

class NgravPacket:
    """
    Represents a Ngrav model package.

    Responsibilities of this class for now:

    1. Accept an ONNX model.
    2. Verify that it is actually a valid ONNX model.
    3. Copy it into the Ngrav package structure.
    4. Extract a small amount of ONNX metadata.
    5. Store that metadata in manifest.yaml.
    6. Re-open an existing Ngrav package.
    """

    __slots__ = ("package_path", "_title", "_model", "_manifest", "_fingerprint")

    def __init__(self, package_path: PathInput, title: str | None):
        if not isinstance(package_path, PathInput):
            raise TypeError(f"Package path must be Type: PathInput - Received Dtype: {type(package_path).__name__}")

        if not isinstance(title, str):
             raise TypeError(f"Title must be Type: Str - Received Dtype: {type(title).__name__}")
        
        self.package_path = Path(package_path)
        self._title = package_path if title is None else title
        self._fingerprint = None
        self._model = None
        self._manifest = None

    #==================================================================================================================
    # NGRAV PACKET: Properties
    #==================================================================================================================

    @property
    def title(self) -> str:
        """
        String title attached to the Ngrav Packet-object
         
        ----- Returns -----
        Onnx.ModelProto
             
        """
        return self._title
    
    @property
    def model(self) -> ModelProto:
        """
        Reference to the internal ONNX model.

        ----- Returns -----
        Onnx.ModelProto
    
        """

        return self._model

    @property
    def manifest(self) -> str:
        """
        Copy of the internal metadata related to ONNX model
        
        ----- Returns -----
        YAML.file
            
        """

        return self._manifest

    @property
    def model_path(self) -> Path:
        """
        Path representation of the initial ONNX model filepath.
        
        ----- Returns -----
        pathlib.Path
            
        """

        return self.package_path

    #==================================================================================================================
    # NGRAV PACKET: Class Methods
    #==================================================================================================================

    @classmethod
    def from_onnx(cls, model_path: PathInput, title: str | None) -> NgravPacket:
        """
        Construct an in-memory NgravPackage from an ONNX file.

        Nothing is written to disk.
        """

        # Validate parameters
        if not isinstance(model_path, PathInput):
            raise TypeError(f"Model path should be Type: PathInput - Received dtype: {type(model_path).__name__}")

        if not isinstance(title, str):
            raise TypeError(f"Title should be Type: Str - Received dtype: {type(title).__name__}")

        final_path = Path(model_path)   # Convert the parameter input to valid Path-object

        # Validate Path-object - File exists, is file, file is ONNX format
        if not final_path.exists():
            raise FileNotFoundError(f"Requested ONNX model file doesn't not exist - {final_path}")

        if not final_path.is_file():
                    raise ValueError(f"Object received is not a valid file format - {final_path}")

        if final_path.suffix.lower() != ".onnx":
            raise ValueError(f"Requested model file is not a valid ONNX format - {final_path}")

        # Load onnx model from vaild Path-object and check model integrity
        try:
            model = onnx.load(final_path)
            onnx.checker.check_model(model)
        except Exception:
            raise InvalidOnnxModelError(f"Failed to load ONNX model")

        # Build initial mmetadata manifest
        manifest = cls._build_manifest(
            model=model,
            model_source=final_path
        )

        fingerprint = cls._construct_fingerprint()

        # Create and return new NgravPacket object
        return cls(
            _model=model,
            _manifest=manifest,
            _fingerprint=fingerprint
        )

    #==================================================================================================================
    # NGRAV PACKET: Static Methods
    #==================================================================================================================

    @staticmethod
    def _build_manifest(model_source: Path, model: ModelProto) -> NgravManifest:
        """
        Extract the first small set of reproducibility metadata.

        This method will naturally grow as Ngrav's manifest evolves.
        """

        opoperator_sets = [
            BaseInfo(
                domain=opset.domain,
                version=opset.version
            )
            for opset in model.opset_import
        ]

        onnx_info = OnnxModelInfo(
            ir_version=model.ir_version,
            producer_name=model.producer_name or None,
            producer_version=model.producer_version or None,
            model_version=model.model_version or None,
            graph_name=model.graph.name or None,
            opsets=opoperator_sets,
        )

        return NgravManifest(
            name=model_source.stem,
            original_filename=model_source.name,
            valid_onnx=True,
            onnx=onnx_info,
        )

    @staticmethod
    def _construct_fingerprint() -> None:
        return ""

    #==================================================================================================================
    # NGRAV PACKET: Magic Methods
    #==================================================================================================================

    def __repr__(self) -> str:
        return f"Ngrav Package:" 

    def __eq__(self, other: NgravPacket) -> bool:
        return self._fingerprint == other._fingerprint
        