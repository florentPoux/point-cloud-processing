# Spatial AI Studio - Premium GUI

## Overview

Spatial AI Studio is a premium dark-mode GUI application for running point cloud processing tutorials with **automatic visualization generation** and **article-ready output structuring**.

## Features

### 🎨 Premium Dark Mode Interface
- Professional dark theme optimized for long work sessions
- PyQt5-powered responsive UI
- Clean, modern design with intuitive controls

### 🔬 Interactive 3D Visualization
- **PyVista integration** for high-quality 3D rendering
- Real-time point cloud visualization
- Interactive camera controls
- Automatic 45° elevation viewing angle

### 📊 Automatic Plot Generation
- Step-by-step visualization capture
- 2D matplotlib charts for metrics
- High-resolution exports (150 DPI)
- Dark-themed consistent styling

### 🎬 Rotation GIF Generation
- **Automatic 360° rotation GIFs**
- **Fixed 45° elevation angle** for professional appearance
- Configurable frame count (12-72 frames)
- Optimized file size with imageio

### 📁 Article-Ready Output Structure
```
tutorial_outputs/
├── System_02_-_Gaussian_Splatting/
│   ├── step_01.png
│   ├── step_02.png
│   ├── ...
│   ├── step_09.png
│   ├── rotation_45deg.gif
│   └── metadata.json
├── System_03_-_Smart_Point_Cloud/
│   └── ...
└── ...
```

### 🚀 Tutorial Execution
- Run individual tutorials or all sequentially
- Background thread execution (UI stays responsive)
- Real-time progress tracking
- Comprehensive execution logs

### 📈 Integrated Systems

The GUI supports all 9 systems with comprehensive step-by-step tutorials:

1. **System 02 - Gaussian Splatting** (9 steps)
   - Novel view synthesis pipeline
   - 3D Gaussian initialization to rendering

2. **System 03 - Smart Point Cloud** (9 steps)
   - Memory-mapped processing
   - Filtering, segmentation, registration

3. **System 04 - GeoAI & Mapping** (9 steps)
   - DEM/DSM generation
   - Web tile creation with Leaflet

4. **System 05 - Deep Learning 3D** (9 steps)
   - PointNet training pipeline
   - Classification and prediction

5. **System 06 - Zero-Shot Foundation** (9 steps)
   - CLIP and SAM integration
   - Multi-view rendering

6. **System 07 - Spatial Agentic AI** (9 steps)
   - Autonomous task planning
   - Tool execution and reflection

7. **System 08 - Scan-to-BIM** (9 steps)
   - Building element extraction
   - IFC and DXF export

8. **System 09 - Digital Twin** (9 steps)
   - Real-time synchronization
   - IoT sensor integration

## Installation

### Requirements

```bash
pip install -r requirements_gui.txt
```

### Required Packages
- PyQt5 >= 5.15.0
- pyvistaqt >= 0.9.0
- pyvista >= 0.38.0
- matplotlib >= 3.5.0
- imageio >= 2.19.0
- numpy, scipy (core dependencies)

## Usage

### Launch the GUI

```bash
python spatial_ai_studio.py
```

### Running a Tutorial

1. **Select Tutorial** from dropdown menu
2. **Configure Options**:
   - Auto-export plots ✓
   - Generate rotation GIF (45°) ✓
   - Structure for article automation ✓
   - Set GIF frame count (default: 36)
3. **Select Output Directory** (optional)
4. **Click "▶ Run Tutorial"**

### Running All Tutorials

Click **"▶▶ Run All Tutorials"** to execute all 9 systems sequentially with automatic visualization generation.

### Visualization Tabs

- **3D Viewer**: Interactive PyVista visualization
- **Plots & Charts**: 2D matplotlib metrics
- **Execution Log**: Real-time progress and messages

## Output Structure for Article Automation

Each tutorial generates:

### 1. Step-by-Step Visualizations
- `step_01.png` through `step_09.png`
- High-resolution (150 DPI)
- Consistent styling (dark background)
- 45° viewing angle for 3D plots

### 2. Rotation GIF
- `rotation_45deg.gif`
- 360° azimuth rotation
- Fixed 45° elevation
- Configurable frame count
- Optimized for web embedding

### 3. Metadata JSON
```json
{
  "tutorial_name": "System 02 - Gaussian Splatting",
  "execution_time": "2024-01-14T10:30:45",
  "output_directory": "./tutorial_outputs/System_02_-_Gaussian_Splatting",
  "steps": [
    {
      "step_number": 1,
      "step_name": "Load point cloud data",
      "visualization": "step_01.png"
    },
    ...
  ],
  "rotation_gif": "rotation_45deg.gif"
}
```

## Integrating with Article Generation

The structured output enables **automatic article creation**:

```python
import json
from pathlib import Path

# Load metadata
with open('tutorial_outputs/System_02_.../metadata.json') as f:
    meta = json.load(f)

# Generate article sections
for step in meta['steps']:
    print(f"## Step {step['step_number']}: {step['step_name']}")
    print(f"![Step {step['step_number']}]({step['visualization']})")
    print()

# Add rotation GIF
print(f"## 3D Visualization")
print(f"![Rotation]({meta['rotation_gif']})")
```

## Keyboard Shortcuts

- **3D Viewer**:
  - Mouse drag: Rotate view
  - Scroll: Zoom in/out
  - Right-click drag: Pan
  - `r`: Reset camera
  - `s`: Surface representation
  - `w`: Wireframe representation

## Customization

### Modify Tutorial Steps

Edit `TUTORIALS` dictionary in `spatial_ai_studio.py`:

```python
TUTORIALS = {
    'My Custom Tutorial': {
        'module': 'my_module',
        'function': 'my_demo_function',
        'description': 'Custom tutorial description',
        'steps': [
            'Step 1 description',
            'Step 2 description',
            ...
        ]
    }
}
```

### Adjust Visualization Settings

```python
# In create_rotation_gif method
n_frames = 36  # Change frame count
elev = 45      # Change elevation angle
azim = ...     # Azimuth changes per frame
```

## Performance Tips

1. **Large Datasets**: GUI automatically downsamples to 5000 points for visualization
2. **Memory Usage**: Tutorial execution runs in separate thread
3. **GIF Generation**: Higher frame counts increase file size and generation time
4. **Output Directory**: Use SSD for faster I/O during batch processing

## Troubleshooting

### PyVista Not Rendering

```bash
# Install VTK properly
pip uninstall vtk
pip install vtk>=9.1.0
```

### Qt Platform Plugin Error

```bash
# Set Qt platform
export QT_QPA_PLATFORM=xcb  # Linux
export QT_QPA_PLATFORM=windows  # Windows
```

### Import Errors

```bash
# Reinstall all GUI requirements
pip install -r requirements_gui.txt --force-reinstall
```

## Architecture

### Main Components

1. **SpatialAIStudio**: Main window and orchestrator
2. **TutorialRunner**: Background thread for execution
3. **PyVistaViewer**: 3D visualization widget
4. **MatplotlibCanvas**: 2D plotting widget
5. **Demo Functions**: Synthetic data generation

### Data Flow

```
User Selection → TutorialRunner Thread → Step Execution
                                      ↓
                           Generate Visualizations
                                      ↓
                           Save Structured Output
                                      ↓
                           Update UI (signals)
```

## Future Enhancements

- [ ] Cloud GPU execution for heavy processing
- [ ] Remote tutorial execution via SSH
- [ ] Integration with GPT-4 for automatic article generation
- [ ] VR/AR export for immersive visualization
- [ ] Real-time collaboration features
- [ ] Tutorial recording and playback
- [ ] Custom tutorial builder UI
- [ ] Batch processing job queue

## License

Same as main project (see root LICENSE file)

## Support

For issues related to the GUI:
1. Check this README first
2. Review execution log in GUI
3. Open issue at https://github.com/anthropics/claude-code/issues

---

**Created with ❤️ for The Principal Spatial AI Architect program**

*Empowering automated article generation through structured visualization*
