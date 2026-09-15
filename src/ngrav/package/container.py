#==================================================================================================================
# IMPORTS
#==================================================================================================================

import yaml
import json
import numpy as np

from __future__ import annotations

from typing import Any
from datetime import datetime, UTC
from time import perf_counter_ns
from uuid import UUID, uuid4

from pathlib import Path
from hashlib import sha256

import onnxruntime as ort
from onnx import ModelProto

from ..utility import (PathInput, TensorInput, get_initial_execution_metadata)
from ..exceptions import (InvalidOnnxPackageError, InvalidOnnxModelError, UnsupportedOnnxModelError, NgravExecutionError)

from manifest import (NgravManifest, BaseInfo, OnnxModelInfo)
from fingerprint import construct_architecture_fingerprint

from ngrav.model import OnnxSourceSnapshot
from ngrav.history import ExecutionHistory, ExecutionRecord, describe_tensor_values, describe_runtime_info, describe_execution_error

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

    __slots__ = (
        "_source",
        "_source_path", 
        "_title", 
        "_model", 
        "_manifest", 
        "_fingerprint",
        "_session",
        "_valid_onnx",
        "_history"
    )

    def __init__(
        self, 
        source: OnnxSourceSnapshot, 
        source_path: PathInput, 
        history: ExecutionHistory,
        title: str | None = None,
        *args,
        manifest: NgravManifest | None = None,
        fingerprint: str | None = None,
        valid_onnx: bool = True,
        **kwargs
    ):
        self._source = source
        self._source_path = Path(source_path)
        self._title = str(source_path) if title is None else title
        self._model: ModelProto | None = None
        self._manifest = manifest
        self._fingerprint = fingerprint
        self._valid_onnx = valid_onnx
        self._session: ort.InferenceSession | None = None
        self._history = history

    #==================================================================================================================
    # NGRAV PACKET: Properties
    #==================================================================================================================

    @property
    def title(self) -> str:
        """
        String title attached to the Ngrav Packet-object
         
        ----- Returns -----
        str 
        """
        return self._title
    
    @property
    def model(self) -> ModelProto:
        """
        Reference to the internal ONNX model.

        ----- Returns -----
        Onnx.ModelProto
        """

        return self._get_model()

    @property
    def manifest(self) -> NgravManifest:
        """
        Copy of the internal metadata related to ONNX model
        
        ----- Returns -----
        NgravManifest
        """

        return self._get_manifest()

    @property
    def fingerprint(self) -> str:
        """
        Deterministic Hash-fingerprint based on internal ONNX model architecture.
                 
        ----- Returns -----
        str 
        """

        return self._get_fingerprint()

    @property
    def source_path(self) -> Path:
        """
        Path representation of the initial ONNX model filepath.
        
        ----- Returns -----
        pathlib.Path
        """

        return self._source.source_path

    @property
    def is_loaded(self) -> bool:
        """
        Return True if the internal ONNX model has been loaded into memory.

        Lazy loading validation feature.
                
        ----- Returns -----
        bool
        """

        return self._model is None

    @property
    def history(self) -> ExecutionHistory:
        """
        Returns an overview of the NgravPacke's complete execution history complete with a comprehensive suite of metadata.
                        
        ----- Returns -----
        ngrav.ExecutionHistory
        """
        self._history

    @property
    def last_execution(self) -> ExecutionRecord | None:
        """
        Returns the record of the last execution.
                                
        ----- Returns -----
        ngrav.ExecutionRecord
        """

        if not self._history.records:
            return None

        return self._history.records[-1]
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

        if title is not None and not isinstance(title, str):
            raise TypeError(f"Title should be Type: Str - Received dtype: {type(title).__name__}")

        final_path = Path(model_path)   # Convert the parameter input to valid Path-object

        # Validate Path-object - File exists, is file, file is ONNX format
        if not final_path.exists():
            raise FileNotFoundError(f"Requested ONNX model file doesn't not exist - {final_path}")

        if not final_path.is_file():
                    raise ValueError(f"Object received is not a valid file format - {final_path}")

        if final_path.suffix.lower() != ".onnx":
            raise ValueError(f"Requested model file is not a valid ONNX format - {final_path}")

        # Construct a deterministic ONNX model snapshot for model initialization (Lazy loading feature)
        source = OnnxSourceSnapshot.from_path(filepath=final_path)
        history = ExecutionHistory.create()

        return cls(
            history=history,
            source=source,
            title=title,
            valid_onnx=True
        )

    #==================================================================================================================
    # NGRAV PACKET: Static Methods
    #==================================================================================================================

    @staticmethod
    def _construct_manifest(model_source: Path, model: ModelProto) -> NgravManifest:
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
    def _construct_fingerprint(model: ModelProto) -> bytes:
        return construct_architecture_fingerprint(model)

    #==================================================================================================================
    # NGRAV PACKET: Instance Methods
    #==================================================================================================================

    def run(self, inputs: TensorInput) -> dict[str, Any]:
        """
        Run inference using internal ONNX model.

        Parameters
        ----------
        inputs:
            Mapping of ONNX input values to NumPy arrays.

        Returns
        -------
        dict[str, Any]
            Mapping of ONNX output values to respective results.

        Raises
        ------
        TypeError
            If `inputs` is not a mapping or contains invalid input names.

        NgravExecutionError
            Raised if interal ONNX Runtime inference session cannot execute the model.
        """

        # Get execution metadata - UUID, Start time, NS counter
        execution_id, started_at, started_ns = get_initial_execution_metadata()

        input_info = []
        output_info = []

        try:
            if not isinstance(inputs, TensorInput):
                raise TypeError(f"Inputs must be of Type: TensorInput - Received dtype: {type(inputs).__name__}")

            if not all(isinstance(name, str) for name in inputs):
                raise TypeError(f"ONNX inputs names must be of Type: str")

            input_info = [
                
            ]

            # Retrieve internal ONNX Runtime inferene session - Uses Lazy loading.
            session = self._get_session()

            # Run the actual values through the interna ONNX Runtime inference session.
            try:
                results = session.run(None, dict(inputs))
            except Exception:
                raise NgravExecutionError(f"ONNX model execution failed")

            # Map output values to their respective names retrieved from the inference session.
            output_names = [
                output.name
                for output in session.get_outputs()
            ]

            return dict(
                zip(
                    output_names,
                    results,
                    strict=True
                )
            )
        except Exception:
            pass
    
    def compare(self, other: NgravPacket) -> None:
        pass

    def save_model(self, filepath: PathInput) -> None:
        # Validate 'filepath' parameter dtype.
        if not isinstance(filepath, PathInput):
            raise TypeError(f"Path input must be of Type: PathInput - Received dtype: {type(filepath).__name__}")

        model_path = Path(filepath)   # Convert the parameter input to valid Path-object

        # Validate model path extension uses '.onnx' standard.
        if not model_path.suffix.lower() != ".onnx":
            raise ValueError(f"Model destination must use the '.onnx' file extension: {model_path.parent}")

        # Validate model parent directory exists.
        if not model_path.parent.exists():
            raise FileNotFoundError(f"Destination directory doesn't exist: {model_path.parent}")

        # Validate model parent directory is a valid directory.
        if not model_path.parent.is_dir():
            raise NotADirectoryError(f"Destination parent is not a directory: {model_path.parent}")
        
        # Validate model path references a concrete file.
        if model_path.exists() and not model_path.is_file():
            raise IsADirectoryError(f"Destination path is not a file: {model_path}")

        model_path.write_bytes(self._source.model_bytes)    # Write model bytes to '.onnx' file.

    def save_manifest(self, filepath: PathInput, format: str | None = "yaml") -> None:
        """
        Serialize the packet manifest to YAML or JSON.

        Accessing ``self.manifest`` lazily constructs the manifest when required.
        Saving does not alter or remove the manifest stored by the NgravPacket.

        Parameters
        ----------
        filepath:
            Destination path for the manifest.

        format:
            Serialization format. Supported values are ``"yaml"`` and ``"json"``.
        """
        if not isinstance(filepath, PathInput):
            raise TypeError(f"Path input must be of Type: PathInput - Received dtype: {type(filepath).__name__}")

        if not isinstance(format, str):
            raise TypeError(f"Format must be of Type: Str - Received dtype: {type(format).__name__}")

        normalized_format = format.lower()

        if normalized_format not in {"yaml", "json"}:
            raise ValueError(f"Requested manifest format not supported: {format} - Supported format-types: ['yaml', 'json']")
        
        manifest_path = Path(filepath)   # Convert the parameter input to valid Path-object
                
        if not manifest_path.parent.exists():
            raise FileNotFoundError(f"Destination directory does not exist: {manifest_path.parent}")

        if not manifest_path.parent.is_dir():
            raise NotADirectoryError(f"Destination parent is not a directory: {manifest_path.parent}")

        if manifest_path.exists() and not manifest_path.is_file():
            raise IsADirectoryError(f"Destination path is not a file - {manifest_path}")

        expected_suffix = (".yaml" if normalized_format == "yaml" else ".json")

        if manifest_path.suffix.lower() != expected_suffix:
            raise ValueError(f"{normalized_format.upper()} manifest destination must use the '{expected_suffix}' extension - received: {manifest_path}")

        # Calling the property intentionally triggers lazy manifest
        # construction if it has not already been created.
        manifest_data = self.manifest.model_dump(mode="json")

        # Save internal ONNX model to '.yaml' file.
        if normalized_format == "yaml":
            with manifest_path.open("w", encoding="utf-8") as file:
                yaml.safe_dump(
                    manifest_data,
                    file,
                    sort_keys=False,
                    allow_unicode=True,
                )

            return

        # Save internal ONNX model to '.json' file.
        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(
                manifest_data,
                file,
                indent=2,
                ensure_ascii=False,
            )

            file.write("\n")

    def save_history(self, filepath: PathInput, format: str | None = "yaml") -> None:
        """
        Serialize the packet execution history to YAML or JSON.
        
        Parameters
        ----------
        filepath:
        Destination path for the execution history.
        
        format:
        Serialization format. Supported values are ``"yaml"`` and ``"json"``.
        """
        if not isinstance(filepath, PathInput):
            raise TypeError(f"Path input must be of Type: PathInput - Received dtype: {type(filepath).__name__}")
        
        if not isinstance(format, str):
            raise TypeError(f"Format must be of Type: Str - Received dtype: {type(format).__name__}")
        
        normalized_format = format.lower()
        
        if normalized_format not in {"yaml", "json"}:
            raise ValueError(f"Requested manifest format not supported: {format} - Supported format-types: ['yaml', 'json']")
                
        history_path = Path(filepath)   # Convert the parameter input to valid Path-object
                        
        if not history_path.parent.exists():
            raise FileNotFoundError(f"Destination directory does not exist: {history_path.parent}")
        
        if not history_path.parent.is_dir():
            raise NotADirectoryError(f"Destination parent is not a directory: {history_path.parent}")
        
        if history_path.exists() and not history_path.is_file():
            raise IsADirectoryError(f"Destination path is not a file - {history_path}")
        
        expected_suffix = (".yaml" if normalized_format == "yaml" else ".json")
        
        if history_path.suffix.lower() != expected_suffix:
            raise ValueError(f"{normalized_format.upper()} manifest destination must use the '{expected_suffix}' extension - received: {history_path}")

        # Calling the property intentionally triggers lazy manifest
        # construction if it has not already been created.
        history_data = self.manifest.model_dump(mode="json")
        
        # Save internal NgravPacket execution history to '.yaml' file.
        if normalized_format == "yaml":
            with history_path.open("w", encoding="utf-8") as file:
                yaml.safe_dump(
                    history_data,
                    file,
                    sort_keys=False,
                    allow_unicode=True,
                )
        
            return
        
        # Save internal NgravPacket execution history to '.json' file.
        with history_path.open("w", encoding="utf-8") as file:
            json.dump(
                history_data,
                file,
                indent=2,
                ensure_ascii=False,
            )
        
            file.write("\n")

    #==================================================================================================================
    # NGRAV PACKET: Helper Functions
    #==================================================================================================================

    def _get_model(self) -> ModelProto:
        # HELPER-METHOD
        # Check if the internal variable '_model' is instantialized - If not, load it into memory (Lazy loading feature).
        if self._model is None:
            self._model = self._source.load_model()

        return self._model

    def _get_manifest(self) -> NgravManifest:
        # HELPER-METHOD
        # Check if the internal variable '_manifest' is instantialized - If not, load it into memory (Lazy loading feature).
        if self._manifest is None:
            self._manifest = self._construct_manifest(self._source.source_path, self._get_model())
    
        return self._manifest

    def _get_session(self) -> ort.InferenceSession:
        # HELPER-METHOD
        # Check if the internal variable '_session' is instantialized - If not, load it into memory (Lazy loading feature).
        if self._session is None:
            try:
                self._session = ort.InferenceSession(self._source.model_bytes)
            except Exception:
                raise NgravExecutionError(f"Failed to initialze ONNX Runtime inference session")

        self._session

    def _get_fingerprint(self) -> str:
        # HELPER-METHOD
        # Check if the internal variable '_fingerprint' is instantialized - If not, load it into memory (Lazy loading feature).
        if self._fingerprint is None:

            fingerprint_digest = self._construct_fingerprint(self._get_model())
            self._fingerprint = f"sha256:{fingerprint_digest.hex()}"
        
        return self._fingerprint

    #==================================================================================================================
    # NGRAV PACKET: Magic Methods
    #==================================================================================================================

    def __repr__(self) -> str:
        """
        Explicit String representation of the NgravPacket-object.

        Intentionally avoids printing the complete ONNX model configuration or manifest metadata.
        """

        fingerprint = (self._fingerprint if self._fingerprint else "not-loaded")

        model_state = ("loaded" if self._model is not None else "not-loaded")

        return (
            f"<{self.__class__.__name__}>"
            f"title={self._title!r}, "
            f"source_path={str(self._source_path)!r}, "
            f"model={model_state!r}, "
            f"fingerprint={fingerprint!r}"
    )

    def __eq__(self, other: object) -> bool:
        """
        Compare two NgravPacket objects using deterministic fingerprints.

        Packets without concrete fingerprints are not considered equal unless the exact same Python object.
        """

        if not isinstance(other, NgravPacket):
            return False

        if (self._fingerprint is None or other._fingerprint is None):
            return self is other

        return self._fingerprint == other._fingerprint
        