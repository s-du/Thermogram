# App Structure (Thermogram / DroneIR Toolkit)

## Overview
This repository is a **desktop GUI application** (PyQt6) for processing DJI thermal imagery. The app loads a DJI SD-card-like folder structure, extracts radiometric thermal data via DJI tooling, provides interactive visualization/measurements, and supports batch export and Word report generation.

## Top-level entry points
- **`main.py`**
  - Main GUI application entry point.
  - Defines `DroneIrWindow(QMainWindow)` which wires UI actions to processing/export/report functionality.
  - Orchestrates loading datasets, maintaining the current `ProcessedIm`, and updating the viewers.

- **`main.spec`**
  - PyInstaller spec (build packaging).

- **`dialogs.py`**
  - GUI dialogs (measurement dialogs, fusion/overlay dialogs, hotspot detection, alignment UI, etc.).
  - Many dialogs delegate heavy image work to `tools.thermal_tools`.

- **`widgets.py`**
  - Custom PyQt6 widgets used by the main window and dialogs.
  - Key widgets:
    - `PhotoViewer`: main image canvas that supports drawing measurements and showing overlays.
    - `DualViewer`: side-by-side synchronized viewer.
    - `QRangeSlider`: double-ended slider used for temperature range selection.

## Core packages

### `tools/` (processing + core domain objects)
This folder contains most of the application’s “business logic”.

- **`tools/core.py`**
  - Core data types and low-level helpers.
  - Key responsibilities:
    - DJI thermal raw extraction (`read_dji_image`, `extract_raw_data`).
    - Camera undistortion (`CameraUndistorter`).
    - Drone-specific calibration/offset parameters (`DroneModel`).
    - Central image object storing state + annotations: `ProcessedIm`.
    - Measurement objects:
      - `PointMeas`: single temperature spot.
      - `LineMeas`: line profile + extrema.
      - `RectMeas`: ROI statistics (min/max/mean/std, area) and markers.

- **`tools/thermal_tools.py`**
  - High-level thermal processing functions and export routines.
  - Key responsibilities:
    - Palette/colormap registry (DJI-like palettes + custom palettes).
    - Edge overlay generation from RGB into thermal images.
    - Converting thermal float data to colored renderings (`process_raw_data`).
    - Batch export worker classes (`RunnerDJI`, `RunnerMiniature`) using `QThreadPool` / `QRunnable` patterns.

- **`tools/report_tools.py`**
  - Word report generation using `python-docx`.
  - Adds images and measurement summaries (tables) into structured sections.

- **`tools/thermal_3d.py`**
  - 3D visualization utilities (Open3D-based), used for voxel/surface-like views.

- **`tools/thermal_parser.py`**
  - Alternative/legacy thermal parsing logic (derived from an external project).
  - Includes DJI/FLIR parsing helpers and SDK bindings.

- **`tools/ai_tools.py`**
  - AI-related helpers (e.g., object detection/segmentation integrations), consumed from dialogs and/or the main window.

### `utils/` (config + logging + exceptions)
- **`utils/config.py`**
  - Strongly-typed app settings via Pydantic:
    - `AppConfig`: app constants (folder names, UI paths, view/legend labels).
    - `ThermalConfig`: default radiometric parameters and edge defaults.

- **`utils/logger.py`**
  - Logging helpers (`info`, `debug`, `warning`, `error`).

- **`utils/exceptions.py`**
  - App-specific exceptions (e.g., `ThermogramError`, `FileOperationError`).

### `ui/` (Qt Designer files)
- `.ui` files loaded at runtime by `PyQt6.uic.loadUi`.
- `main_window.ui` is the primary window layout.
- Dialog UIs (export dialog, fusion dialog, measurement dialogs, etc.) are also stored here.

### `resources/` (embedded binaries + images + calibration)
- Accessed via `resources.find(rel_path)`.
- Contains:
  - App images/icons.
  - Calibration XMLs used for undistortion.
  - External tooling (e.g., DJI thermal SDK executable, exiftool) that the app calls via subprocess.

## Runtime data flow (high level)

### 1) Load dataset
- Triggered from `DroneIrWindow.load_folder_phase1` (via menu action).
- The app enumerates image files and builds lists of RGB / IR paths.

### 2) Build per-image objects
- For each IR image, a `tools.core.ProcessedIm` is created.
- `ProcessedIm` maintains:
  - Paths (IR, RGB, cropped RGB, original RGB).
  - Radiometric parameters (`thermal_param`).
  - Raw temperature data (`raw_data`, `raw_data_undis`).
  - Display parameters (palette, shown range, post-processing).
  - User measurements (rect/line/point lists) and report flags.

### 3) Extract radiometric data
- `tools.core.extract_raw_data()` calls the DJI tooling (`read_dji_image()`) to produce a temporary `.raw`, reads it into a float32 matrix, undistorts it, then deletes the `.raw`.

### 4) Visualization
- `tools.thermal_tools.process_raw_data()` maps float temperatures to a colormap.
- `widgets.PhotoViewer` displays the resulting pixmap and manages measurement drawing.
- Optional overlays (edge overlays, picture-in-picture) are composed in `tools.thermal_tools` and/or dialogs.

### 5) Measurements
- The user draws measurements in `PhotoViewer`.
- The main window converts these into measurement objects:
  - `PointMeas`, `LineMeas`, `RectMeas`
- Measurements are stored on the active `ProcessedIm` instance and displayed in the tree view.

### 6) Exporting and reporting
- Batch exports use `RunnerDJI` to process multiple images in background threads.
- Reports are generated by `tools.report_tools.create_word_report()` using the current `ProcessedIm` list.

## Conventions / important notes
- **External binaries:** thermal extraction relies on tools located under `resources/` (invoked via `subprocess`).
- **UI loading:** `.ui` files are loaded dynamically at runtime; missing files will raise `FileOperationError`.
- **State ownership:** `DroneIrWindow` owns the active dataset list (`self.images`) and active image index; each `ProcessedIm` owns its own measurements/settings.

## Tests
There are currently ad-hoc test scripts (e.g., `test_*.py`) in the repo root. These are intended to be moved into a dedicated `tests/` folder.
