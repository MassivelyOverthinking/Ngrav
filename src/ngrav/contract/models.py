#==================================================================================================================
# IMPORTS
#==================================================================================================================

from __future__ import annotations

import numpy as np

from typing import Any

from pydantic import BaseModel, Field

#==================================================================================================================
# MODEL CONTRACTS: Mapping
#==================================================================================================================

numpy_tensor_mapping = {

}

#==================================================================================================================
# MODEL CONTRACTS: Base Models
#==================================================================================================================

class TensorSpec(BaseModel):

    dtype: str | None = None
    shape: list[int] = Field(default_factory=list)
    min_value: float = Field(default=0, ge=0)
    max_value: float

class ModelContract(BaseModel):

    name: str
    description: str | None = None

    inputs: dict[str, TensorSpec] = Field(default_factory=dict)
    outputs: dict[str, TensorSpec] = Field(default_factory=dict)