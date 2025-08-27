class ThermogramError(Exception):
    """Base exception class for IR-Lab application"""
    pass

class ImageProcessingError(ThermogramError):
    """Raised when there's an error processing thermal images"""
    pass

class ConfigurationError(ThermogramError):
    """Raised when there's an error in configuration"""
    pass

class UIError(ThermogramError):
    """Raised when there's an error in the UI"""
    pass

class FileOperationError(ThermogramError):
    """Raised when there's an error with file operations"""
    pass
