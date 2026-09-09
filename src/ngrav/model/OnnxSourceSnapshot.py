#==================================================================================================================
# IMPORTS
#==================================================================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import ModelProto
from onnx import external_data_helper

from ngrav.exceptions import (InvalidOnnxModelError, UnsupportedOnnxModelError)

#==================================================================================================================
# ONNX SOURCE SNAPSHOT
#==================================================================================================================

@dataclass(frozen=True, slots=True)
class OnnxSourceSnapshot:
    """
    Immutable snapshot artifact of ONNX source model.
    """

    source_path: Path
    model_bytes: bytes

    @classmethod
    def from_path(cls, filepath: Path) -> OnnxSourceSnapshot:
        try:
            model_bytes = filepath.read_bytes()
        except OSError:
            raise InvalidOnnxModelError(f"Failed to read ONNX file: {filepath}")

        snapshot = cls(
            source_path=filepath,
            model_bytes=model_bytes,
        )

        snapshot.validate()

        return snapshot

    def validate(self) -> None:
        """
        Validate internal model bytes relate to actual ONNX model.

        Parsed ONNX model is only kept temporarily and is dropped shortly after construction.
        """

        try:
            model = onnx.load_model_from_string(self.model_bytes)
        except Exception:
            raise InvalidOnnxModelError(f"Failed to deserialize ONNX model: {self.source_path}")

        if _model_uses_external_data(model=model):
            raise UnsupportedOnnxModelError(f"External tensor data is not supported by current Ngrav configuration")

        try:
            onnx.checker.check_model(model)
        except Exception:
            raise InvalidOnnxModelError(f"ONNX model validation failed: {self.source_path}")

    def load_model(self) -> ModelProto:
        """
        Materialize a concrete ONNX model from internal model bytes.
        """

        try:
            return onnx.load_model_from_string(self.model_bytes)
        except Exception:
            raise InvalidOnnxModelError(f"Failed to materialize the ONNX model: {self.source_path}")

def _model_uses_external_data(model: ModelProto) -> bool:
    return _graph_uses_external_data(model.graph)


def _graph_uses_external_data(graph: onnx.GraphProto) -> bool:
    for tensor in graph.initializer:
        if external_data_helper.uses_external_data(tensor):
            return True

    for sparse_tensor in graph.sparse_initializer:
        if external_data_helper.uses_external_data(sparse_tensor.values):
            return True

        if external_data_helper.uses_external_data(sparse_tensor.indices):
            return True

    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                if external_data_helper.uses_external_data(attribute.t):
                    return True

            elif attribute.type == onnx.AttributeProto.TENSORS:
                if any(
                    external_data_helper.uses_external_data(tensor)
                    for tensor in attribute.tensors
                ):
                    return True

            elif attribute.type == onnx.AttributeProto.GRAPH:
                if _graph_uses_external_data(attribute.g):
                    return True

            elif attribute.type == onnx.AttributeProto.GRAPHS:
                if any(
                    _graph_uses_external_data(nested_graph)
                    for nested_graph in attribute.graphs
                ):
                    return True

    return False

