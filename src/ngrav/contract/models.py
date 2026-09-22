#==================================================================================================================
# IMPORTS
#==================================================================================================================

from __future__ import annotations

import numpy as np

from typing import Any
from enum import StrEnum

from pydantic import BaseModel, Field
from .tensor import TensorDType

#==================================================================================================================
# MODEL CONTRACTS: Base Models
#==================================================================================================================

class TensorSpec(BaseModel):
    dtype: TensorDType | None = None

    shape: list[int] = Field(default_factory=list)

    min_value: float | None = None
    max_value: float | None = None

    allow_nan: bool = False
    allow_inf: bool = False

class ModelContract(BaseModel):

    name: str
    description: str | None = None

    inputs: dict[str, TensorSpec] = Field(default_factory=dict)
    outputs: dict[str, TensorSpec] = Field(default_factory=dict)

    @staticmethod
    def create(cls) -> ModelContract:
        return cls()

    def add_input_tensor_spec(self, name: str, tensorSpec: TensorSpec) -> None:
        self.inputs[name] = tensorSpec

    def add_output_tensor_spec(self, name: str, tensorSpec: TensorSpec) -> None:
        self.outputs[name] = tensorSpec

#==================================================================================================================
# MODEL CONTRACTS: Result Type
#==================================================================================================================

class ValidationStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"

class TensorValidationResult(BaseModel):
    status: ValidationStatus
    errors: list[str] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status == ValidationStatus.VALID