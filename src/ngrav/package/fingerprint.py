#==================================================================================================================
# IMPORTS
#==================================================================================================================

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256

from google.protobuf.message import Message
from onnx import AttributeProto, FunctionProto, GraphProto, ModelProto, NodeProto
from onnx import SparseTensorProto, TensorProto

#==================================================================================================================
# ENUM VALUES: Tensors & Architecture
#==================================================================================================================

_ARCHITECTURE_FINGERPRINT_VERSION = b"ngrav:architecture:v1\0"

# TensorProto can encode values through several different fields depending
# on dtype and serialization strategy. All of these represent tensor payload,
# not tensor architecture.
_TENSOR_VALUE_FIELDS = (
    "segment",
    "float_data",
    "int32_data",
    "string_data",
    "int64_data",
    "raw_data",
    "double_data",
    "uint64_data",
    "external_data",
    "data_location",
)

#==================================================================================================================
# NGRAV FINGERPRINT METHODS
#==================================================================================================================

def build_architecture_fingerprint_payload(model: ModelProto) -> bytes:
    """
    Return a deterministic serialization of an ONNX model architecture.

    The original ModelProto is never mutated.

    Preserved:
        - IR version
        - opset imports
        - graph input/output contracts
        - tensor shapes and dtypes
        - graph connectivity
        - operator types and domains
        - operator attributes
        - initializer names, shapes, and dtypes
        - local ONNX functions
        - nested subgraphs

    Removed:
        - initializer values / learned weights
        - external tensor locations
        - producer metadata
        - model-level descriptive version metadata
        - documentation strings
        - graph/node display names
        - arbitrary model metadata properties
        - training metadata
    """
    canonical_model = ModelProto()
    canonical_model.CopyFrom(model)

    _clear_supported_fields(
        canonical_model,
        "producer_name",
        "producer_version",
        "domain",
        "model_version",
        "doc_string",
        "metadata_props",
        "training_info",
    )

    _canonicalize_opset_imports(canonical_model)
    _canonicalize_graph(canonical_model.graph)

    for function in canonical_model.functions:
        _canonicalize_function(function)

    # Function declaration order is not part of the computational identity.
    _sort_message_field(
        canonical_model,
        "functions",
        key=_deterministic_message_bytes,
    )

    return canonical_model.SerializeToString(
        deterministic=True,
    )


def construct_architecture_fingerprint(model: ModelProto) -> bytes:
    """
    Construct the SHA-256 architecture fingerprint for an ONNX model.

    A version prefix is included so future canonicalization changes can
    deliberately introduce a new fingerprint schema without silently
    changing the meaning of existing fingerprints.
    """
    payload = build_architecture_fingerprint_payload(model)

    return sha256(_ARCHITECTURE_FINGERPRINT_VERSION + payload).digest()


def _canonicalize_graph(graph: GraphProto) -> None:
    _clear_supported_fields(
        graph,
        "name",
        "doc_string",
        "metadata_props",
    )

    for value_info in graph.input:
        _clear_supported_fields(
            value_info,
            "doc_string",
            "metadata_props",
        )

    for value_info in graph.output:
        _clear_supported_fields(
            value_info,
            "doc_string",
            "metadata_props",
        )

    for value_info in graph.value_info:
        _clear_supported_fields(
            value_info,
            "doc_string",
            "metadata_props",
        )

    for initializer in graph.initializer:
        _strip_initializer_values(initializer)

    for sparse_initializer in graph.sparse_initializer:
        _strip_sparse_initializer_values(sparse_initializer)

    for node in graph.node:
        _canonicalize_node(node)

    # These collections do not carry meaningful declaration ordering.
    _sort_message_field(
        graph,
        "initializer",
        key=lambda tensor: (
            tensor.name,
            _deterministic_message_bytes(tensor),
        ),
    )

    _sort_message_field(
        graph,
        "sparse_initializer",
        key=_deterministic_message_bytes,
    )

    _sort_message_field(
        graph,
        "value_info",
        key=lambda value_info: (
            value_info.name,
            _deterministic_message_bytes(value_info),
        ),
    )

    # Node ordering is not treated as architectural identity here.
    # Connectivity is already encoded through node inputs/outputs.
    #
    # Sorting makes independently serialized but structurally identical
    # graphs stable when valid nodes have a different declaration order.
    _sort_message_field(
        graph,
        "node",
        key=_deterministic_message_bytes,
    )


def _canonicalize_node(node: NodeProto) -> None:
    _clear_supported_fields(
        node,
        "name",
        "doc_string",
        "metadata_props",
    )

    for attribute in node.attribute:
        _canonicalize_attribute(attribute)

    # Attribute ordering is semantically irrelevant.
    _sort_message_field(
        node,
        "attribute",
        key=lambda attribute: (
            attribute.name,
            _deterministic_message_bytes(attribute),
        ),
    )


def _canonicalize_attribute(attribute: AttributeProto) -> None:
    _clear_supported_fields(
        attribute,
        "doc_string",
    )

    if attribute.type == AttributeProto.GRAPH:
        _canonicalize_graph(attribute.g)

    elif attribute.type == AttributeProto.GRAPHS:
        for graph in attribute.graphs:
            _canonicalize_graph(graph)

        _sort_message_field(
            attribute,
            "graphs",
            key=_deterministic_message_bytes,
        )

    # Tensor-valued attributes are deliberately NOT stripped here.
    #
    # Unlike graph initializers, tensors embedded in operator attributes
    # may represent inference-relevant constants. Removing them could make
    # computationally different graphs receive the same fingerprint.


def _canonicalize_function(function: FunctionProto) -> None:
    _clear_supported_fields(
        function,
        "doc_string",
        "metadata_props",
    )

    _canonicalize_opset_imports(function)

    for node in function.node:
        _canonicalize_node(node)

    _sort_message_field(
        function,
        "node",
        key=_deterministic_message_bytes,
    )


def _strip_initializer_values(tensor: TensorProto) -> None:
    """
    Remove tensor payload while retaining architecture metadata.

    Fields such as:
        - name
        - dims
        - data_type

    remain intact.
    """
    _clear_supported_fields(
        tensor,
        "doc_string",
        "metadata_props",
        *_TENSOR_VALUE_FIELDS,
    )


def _strip_sparse_initializer_values(
    tensor: SparseTensorProto,
) -> None:
    _strip_initializer_values(tensor.values)
    _strip_initializer_values(tensor.indices)


def _canonicalize_opset_imports(message: Message) -> None:
    if "opset_import" not in message.DESCRIPTOR.fields_by_name:
        return

    _sort_message_field(
        message,
        "opset_import",
        key=lambda opset: (
            opset.domain,
            opset.version,
        ),
    )


def _sort_message_field(
    message: Message,
    field_name: str,
    *,
    key: Callable[[Message], object],
) -> None:
    """
    Sort a repeated protobuf message field without depending on container
    implementation-specific sorting behavior.
    """
    if field_name not in message.DESCRIPTOR.fields_by_name:
        return

    values = list(getattr(message, field_name))

    if len(values) < 2:
        return

    values.sort(key=key)

    message.ClearField(field_name)
    container = getattr(message, field_name)

    for value in values:
        item = container.add()
        item.CopyFrom(value)


def _clear_supported_fields(
    message: Message,
    *field_names: str,
) -> None:
    """
    Clear fields only when present in the installed protobuf schema.

    This avoids tightly coupling Ngrav to fields introduced in one specific
    ONNX protobuf version.
    """
    available_fields = message.DESCRIPTOR.fields_by_name

    for field_name in field_names:
        if field_name in available_fields:
            message.ClearField(field_name)


def _deterministic_message_bytes(message: Message) -> bytes:
    return message.SerializeToString(deterministic=True)