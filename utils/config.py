from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Optional


class ThermalConfig(BaseModel):
    """Thermal processing configuration settings"""
    # Drone model paths
    M2EA_RGB_XML: Path = Path('resources/calibration/m2ea_rgb.xml')
    M2EA_IR_XML: Path = Path('resources/calibration/m2ea_ir.xml')
    M2EA_SAMPLE_RGB: Path = Path('resources/samples/m2ea_rgb.jpg')
    M2EA_SAMPLE_IR: Path = Path('resources/samples/m2ea_ir.jpg')

    M3T_RGB_XML: Path = Path('resources/calibration/m3t_rgb.xml')
    M3T_IR_XML: Path = Path('resources/calibration/m3t_ir.xml')
    M3T_SAMPLE_RGB: Path = Path('resources/samples/m3t_rgb.jpg')
    M3T_SAMPLE_IR: Path = Path('resources/samples/m3t_ir.jpg')

    M30T_RGB_XML: Path = Path('resources/calibration/m30t_rgb.xml')
    M30T_IR_XML: Path = Path('resources/calibration/m30t_ir.xml')
    M30T_SAMPLE_RGB: Path = Path('resources/samples/m30t_rgb.jpg')
    M30T_SAMPLE_IR: Path = Path('resources/samples/m30t_ir.jpg')

    # Default thermal parameters
    DEFAULT_EMISSIVITY: float = 0.95
    DEFAULT_DISTANCE: float = 5.0
    DEFAULT_HUMIDITY: float = 50.0
    DEFAULT_REFLECTION: float = 25.0

    # Image processing settings
    DEFAULT_COLORMAP: str = 'inferno'
    N_COLORS: int = 256
    EDGE_COLOR: str = 'white'
    EDGE_OPACITY: float = 0.7


class AppConfig(BaseModel):
    """Application configuration settings"""
    APP_VERSION: str = '0.1.0'
    APP_FOLDER: str = 'ThermogramApp_'
    ORIGIN_THERMAL_IMAGES_NAME: str = 'Original Thermal Images'
    RGB_ORIGINAL_NAME: str = 'Original RGB Images'
    RGB_CROPPED_NAME: str = 'Cropped RGB Images'
    ORIGIN_TH_FOLDER: str = 'img_th_original'
    RGB_CROPPED_FOLDER: str = 'img_rgb'
    PROC_TH_FOLDER: str = 'img_th_processed'

    RECT_MEAS_NAME: str = 'Rectangle measurements'
    POINT_MEAS_NAME: str = 'Spot measurements'
    LINE_MEAS_NAME: str = 'Line measurements'

    VIEWS: List[str] = ['th. undistorted', 'RGB crop']

    # File paths
    BASE_DIR: Path = Path(__file__).parent.parent
    RESOURCES_DIR: Path = BASE_DIR / 'resources'
    UI_DIR: Path = BASE_DIR / 'ui'

    # Logging configuration
    LOG_LEVEL: str = 'INFO'
    LOG_FILE: Path = BASE_DIR / 'thermogram.log'


# Create global config instances
config = AppConfig()
thermal_config = ThermalConfig()
