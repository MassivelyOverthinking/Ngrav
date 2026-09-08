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
    NgravManifest
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

    __slots__ = ("package_path", "_model", "_manifest")

    MANIFEST_FILENAME = "manifest.yaml"
    MODEL_DIRECTORY = "model"
    MODEL_FILENAME = "model.onnx"

    def __init__(self, package_path: PathInput):
        if not isinstance(package_path, PathInput):
            raise TypeError(f"Package path must be Type: PathInput - Received Dtype: {type(package_path).__name__}")
        
        self.package_path = Path(package_path)
        self._model = None
        self._manifest = None

    @classmethod
    def from_onnx(cls, model_path: PathInput) -> NgravPacket:
        """
        Construct an in-memory NgravPackage from an ONNX file.

        Nothing is written to disk.
        """

        # Validate parameters
        if not isinstance(model_path, PathInput):
            raise TypeError(f"Model path should be Type: PathInput - Received dtype: {type(model_path).__name__}")

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