# 🏗️ Scan-to-BIM & CAD
## System 08: Convert Point Clouds to Building Information Models

Production-ready library for extracting building elements from point clouds and generating BIM models, CAD drawings, and IFC files.

Part of the **Spatial AI Architect Program** by [3D Geodata Academy](https://learngeodata.eu).

## 🎯 Key Features

- **Element Detection**: Automatic detection of walls, floors, doors, windows, columns
- **BIM Generation**: Create parametric building models
- **CAD Export**: DXF export for 2D drawings
- **IFC Generation**: Industry Foundation Classes for BIM interoperability
- **Floor Plans**: Automated 2D floor plan generation
- **Quality Control**: Element validation and quality metrics

## 📦 Installation

```bash
# Install dependencies
pip install numpy scipy scikit-image matplotlib

# Optional: IFC support
pip install ifcopenshell

# Optional: DXF support
pip install ezdxf

# Clone and use
git clone https://github.com/florentPoux/point-cloud-processing.git
cd point-cloud-processing
```

## 🚀 Quick Start

### Complete Scan-to-BIM Pipeline

```python
from scan_to_bim.extraction.element_detector import extract_building_elements
from scan_to_bim.modeling.bim_generator import generate_bim_model
from scan_to_bim.cad_export.dxf_export import export_to_dxf, export_floor_plan
from scan_to_bim.ifc.ifc_generator import export_to_ifc

# Extract elements
elements = extract_building_elements(points, colors)

# Generate BIM model
bim_model = generate_bim_model(elements)

# Export to various formats
export_to_dxf(bim_model, 'building.dxf')
export_floor_plan(bim_model, 'floor_plan.png')
export_to_ifc(bim_model, 'building.ifc', project_name='My Building')

print(f"Extracted {bim_model['metadata']['num_walls']} walls")
print(f"Total floor area: {bim_model['metadata']['total_floor_area']:.2f} m²")
```

## 📚 Modules

### Extraction (`extraction/`)
- `element_detector.py` - Detect walls, floors, doors, windows, columns, stairs

### Modeling (`modeling/`)
- `bim_generator.py` - Generate BIM geometry from detected elements

### CAD Export (`cad_export/`)
- `dxf_export.py` - Export to DXF and generate floor plans

### IFC (`ifc/`)
- `ifc_generator.py` - Export to IFC format for BIM software

## 🎓 Examples

### Element Detection

```python
from scan_to_bim.extraction.element_detector import (
    detect_planes,
    classify_planes_as_elements,
    detect_doors_windows,
    detect_columns
)

# Detect planar surfaces
planes = detect_planes(points, distance_threshold=0.05)

# Classify as building elements
elements = classify_planes_as_elements(planes)

# Separate by type
walls = [e for e in elements if e['element_type'] == 'wall']
floors = [e for e in elements if e['element_type'] == 'floor']

print(f"Detected {len(walls)} walls and {len(floors)} floors")
```

### Wall Boundary Extraction

```python
from scan_to_bim.extraction.element_detector import extract_wall_boundaries

# Extract wall boundaries
boundaries = extract_wall_boundaries(wall_points, resolution=0.1)

for i, boundary in enumerate(boundaries):
    print(f"Wall {i} boundary: {len(boundary)} points")
```

### Door and Window Detection

```python
from scan_to_bim.extraction.element_detector import detect_doors_windows

# Detect openings in wall
openings = detect_doors_windows(wall_points, wall_normal)

doors = [o for o in openings if o['type'] == 'door']
windows = [o for o in openings if o['type'] == 'window']

print(f"Found {len(doors)} doors and {len(windows)} windows")
```

### BIM Model Generation

```python
from scan_to_bim.modeling.bim_generator import (
    create_wall_geometry,
    create_door_geometry,
    create_column_geometry
)

# Create wall geometry
wall_geom = create_wall_geometry(wall_element, thickness=0.2, height=3.0)

# Create door geometry
door_geom = create_door_geometry(door_opening)

# Create column geometry
column_geom = create_column_geometry(column_element, subdivisions=16)
```

### CAD Export

```python
from scan_to_bim.cad_export.dxf_export import export_to_dxf, export_floor_plan

# Export to DXF
export_to_dxf(bim_model, 'building.dxf', layer_prefix='BUILDING')

# Generate floor plan visualization
export_floor_plan(bim_model, 'floor_plan.png', scale=100)
```

### IFC Export

```python
from scan_to_bim.ifc.ifc_generator import export_to_ifc, create_ifc_metadata

# Export to IFC
export_to_ifc(bim_model, 'building.ifc', project_name='Scanned Building')

# Create metadata
metadata = create_ifc_metadata(bim_model)
print(f"IFC Schema: {metadata['schema']}")
```

## 📖 Resources

- [IFC Documentation](https://www.buildingsmart.org/)
- [3D Geodata Academy](https://learngeodata.eu)
- [Scan-to-BIM Tutorial](https://medium.com/@florentpoux)

## 👨‍💻 Author

**Florent Poux** - [3D Geodata Academy](https://learngeodata.eu)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
