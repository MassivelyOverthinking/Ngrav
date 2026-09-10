#==================================================================================================================
# IMPORTS
#==================================================================================================================

from __future__ import annotations

from pathlib import Path
from hashlib import sha256

from onnx import ModelProto

from ..utility import (PathInput)
from ..exceptions import (InvalidOnnxPackageError, InvalidOnnxModelError, UnsupportedOnnxModelError)

from manifest import (NgravManifest, BaseInfo, OnnxModelInfo)
from fingerprint import construct_architecture_fingerprint

from ngrav.model import OnnxSourceSnapshot

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
        "_valid_onnx"
    )

    def __init__(
        self, 
        source: OnnxSourceSnapshot, 
        source_path: PathInput, 
        title: str | None = None,
        *,
        manifest: NgravManifest | None = None,
        fingerprint: str | None = None,
        valid_onnx: bool = True
    ):
        self._source = source
        self._source_path = Path(source_path)
        self._title = str(source_path) if title is None else title
        self._model: ModelProto | None = None
        self._manifest = manifest
        self._fingerprint = fingerprint
        self._valid_onnx = valid_onnx

    #==================================================================================================================
    # NGRAV PACKET: Properties
    #==================================================================================================================

    @property
    def title(self) -> str:
        """
        String title attached to the Ngrav Packet-object
         
        ----- Returns -----
        Str
             
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
        Str 
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

        return cls(
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

    def compare(self, other: NgravPacket) -> None:
        pass

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

    def _get_fingerprint(self) -> str:
        # HELPER-METHOD
        # Check if the internal variable '_fingerprint' is instantialized - If not, load it into memory (Lazy loading feature).
        if self._fingerprint is None:

            fingerprint_digest = self._construct_fingerprint(self._get_model())
            fingerprint_str = f"sha256:{fingerprint_digest.hex()}"

            self._fingerprint = fingerprint_str
        
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
        