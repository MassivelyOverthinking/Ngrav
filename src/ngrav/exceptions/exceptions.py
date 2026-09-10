#==================================================================================================================
# IMPORTS
#==================================================================================================================


#==================================================================================================================
# CUSTOM EXCEPTIONS
#==================================================================================================================

class BaseError(Exception):
    """
    Base exception for all Ngrav-specific errors.

    Parameters
    ----------
    message:
        Human-readable description of what went wrong.
    """
    def __init__(self, message: str) -> None:
        self.messaeg = message
        super().__init__(message)

class InvalidOnnxModelError(BaseError):
    """
    Raised when a supplied file cannot be treated as a valid ONNX model.
    """
    def __init__(self, message: str) -> None:
        super().__init__(message)

class InvalidOnnxPackageError(BaseError):
    """
    Raised when a directory does not conform to the expected Ngrav package structure.
    """
    def __init__(self, message: str) -> None:
        super().__init__(message)

class UnsupportedOnnxModelError(BaseError):
    """
    Raised when the supplied model is valid ONNX, but uses a feature that the current version of Ngrav does not support.
    """
    def __init__(self, message: str) -> None:
        super().__init__(message)

class NgravExecutionError(BaseError):
    """
    Raised when a ONNX model cannot be poperly executed
    """
    def __init__(self, message: str) -> None:
        super().__init__(message)