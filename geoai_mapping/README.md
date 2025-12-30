# 🌍 GeoAI & Mapping
## System 04: Intelligent Geographic Data Processing

Production-ready library for georeferenced point cloud processing, terrain generation, land cover classification, and web map tile creation. Designed for large-scale mapping applications with efficient coordinate transformations and real-time rasterization.

Part of the **Spatial AI Architect Program** by [3D Geodata Academy](https://learngeodata.eu).

## 🎯 Key Features

- **Coordinate Systems**: Transform between EPSG codes, UTM, WGS84, local tangent planes
- **Terrain Generation**: DEM, DSM, DTM, CHM with hydrological analysis
- **Land Cover Classification**: ML-based classification with geometric and spectral features
- **Change Detection**: Multi-temporal analysis for urban monitoring
- **Orthophoto Generation**: High-quality orthophotos with mosaicking and blending
- **Web Map Tiles**: XYZ/TMS tile generation with Leaflet viewer
- **Mesh Generation**: Delaunay triangulation with texture mapping

## 📦 Installation

```bash
# Install dependencies
pip install numpy scipy scikit-learn scikit-image laspy pyproj Pillow

# Optional: GDAL for advanced GIS support
pip install gdal

# Clone and use
git clone https://github.com/florentPoux/point-cloud-processing.git
cd point-cloud-processing
```

## 🚀 Quick Start

### Complete Mapping Workflow

```bash
python geoai_mapping/examples/complete_mapping_workflow.py input.las --output ./maps --resolution 0.5
```

### Python API

```python
from geoai_mapping.terrain.dem_dsm import generate_dsm, generate_dtm, compute_chm
from geoai_mapping.ortho.orthophoto import generate_orthophoto, export_geotiff

# Generate terrain models
dsm, geotransform = generate_dsm(points, resolution=0.5)
dtm, _ = generate_dtm(ground_points, resolution=0.5)
chm = compute_chm(dsm, dtm)

# Generate orthophoto
ortho, ortho_geotrans = generate_orthophoto(points, colors, resolution=0.1)
export_geotiff('orthophoto.tif', ortho, ortho_geotrans, epsg=32632)
```

## 📚 Modules

### Core (`core/`)
- `coordinates.py` - Coordinate system transformations (EPSG, UTM, WGS84, ENU)
- `georef.py` - Georeferencing utilities, worldfiles, spatial indexing

### Terrain (`terrain/`)
- `dem_dsm.py` - DEM/DSM/DTM generation, slope, aspect, hillshade, contours
- `mesh_generation.py` - Delaunay triangulation, mesh simplification, texture mapping

### Classification (`classification/`)
- `land_cover.py` - ML-based land cover classification with Random Forest

### Change Detection (`change_detection/`)
- `temporal.py` - Multi-temporal analysis, building change detection, volumetric changes

### Orthophoto (`ortho/`)
- `orthophoto.py` - Orthophoto generation, mosaicking, color balancing, GeoTIFF export

### Web Tiles (`web_tiles/`)
- `tile_generator.py` - XYZ/TMS tile generation, MBTiles packaging, Leaflet viewer

## 🎓 Examples

### Coordinate Transformation

```python
from geoai_mapping.core.coordinates import transform_coordinates, wgs84_to_utm

# Transform from UTM to WGS84
points_utm = np.array([[500000, 4649776, 100]])
points_wgs84 = transform_coordinates(points_utm, from_epsg=32632, to_epsg=4326)

# Convert WGS84 to UTM
lat_lon = np.array([[-0.001, 51.477, 0]])
utm_coords = wgs84_to_utm(lat_lon, zone=31)
```

### DEM Processing

```python
from geoai_mapping.terrain.dem_dsm import (
    generate_dem, compute_slope, compute_hillshade, extract_contours
)

# Generate DEM with IDW interpolation
dem, geotransform = generate_dem(points, resolution=1.0, method='idw', power=2)

# Compute terrain derivatives
slope = compute_slope(dem, resolution=1.0)
hillshade = compute_hillshade(dem, resolution=1.0, azimuth=315, altitude=45)

# Extract contours
contours = extract_contours(dem, geotransform, interval=10)
```

### Land Cover Classification

```python
from geoai_mapping.classification.land_cover import (
    extract_features_for_classification,
    train_land_cover_classifier,
    create_land_cover_map
)

# Extract features
features = extract_features_for_classification(points, colors, normals, k_neighbors=20)

# Train classifier
classifier, scaler = train_land_cover_classifier(features, labels, n_estimators=100)

# Create land cover map
land_cover_map, geotransform = create_land_cover_map(
    points, predictions, resolution=0.5, num_classes=5
)
```

### Change Detection

```python
from geoai_mapping.change_detection.temporal import (
    detect_changes_dem,
    detect_building_changes,
    compute_volumetric_change
)

# DEM change detection
change_mask, change_mag, change_type = detect_changes_dem(
    dem_2020, dem_2025, threshold=0.5
)

# Building change detection
building_changes = detect_building_changes(
    dsm_2020, dtm_2020, dsm_2025, dtm_2025, height_threshold=3.0
)

# Volumetric analysis
volume_stats = compute_volumetric_change(dem_t1, dem_t2, resolution=0.5)
print(f"Net volume change: {volume_stats['net_volume_m3']:.2f} m³")
```

### Web Map Tiles

```python
from geoai_mapping.web_tiles.tile_generator import (
    generate_tiles_from_image,
    generate_leaflet_viewer
)

# Generate tiles for multiple zoom levels
tiles_generated = generate_tiles_from_image(
    ortho, geotransform, './tiles',
    zoom_levels=[12, 13, 14], epsg=32632
)

# Create interactive viewer
generate_leaflet_viewer('./tiles', {
    'name': 'My Orthophoto',
    'minzoom': 12,
    'maxzoom': 14
})
```

## 🏆 Advanced Features

### Mesh Generation

```python
from geoai_mapping.terrain.mesh_generation import (
    generate_mesh_from_dem,
    compute_vertex_normals,
    export_mesh_obj
)

# Generate mesh from DEM
vertices, faces = generate_mesh_from_dem(dem, geotransform, simplification=2)

# Compute smooth normals
normals = compute_vertex_normals(vertices, faces)

# Export to OBJ
export_mesh_obj('terrain.obj', vertices, faces, normals)
```

### Building Footprint Extraction

```python
from geoai_mapping.terrain.mesh_generation import extract_building_footprints

# Extract building footprints from nDSM
footprints = extract_building_footprints(
    dsm, dtm, min_height=3.0, min_area=20.0, resolution=0.5
)

for building in footprints:
    print(f"Building area: {building['area']:.2f} m²")
```

## 📖 Resources

- [3D Geodata Academy](https://learngeodata.eu)
- [GeoAI Tutorial Series](https://medium.com/@florentpoux)
- [Point Cloud Processing Guide](https://towardsdatascience.com/discover-3d-point-cloud-processing-with-python-6112d9ee38e7)

## 🛠️ Built With

* [Python](https://www.python.org/) - Programming language
* [NumPy](https://numpy.org/) - Array operations
* [SciPy](https://scipy.org/) - Scientific computing
* [scikit-learn](https://scikit-learn.org/) - Machine learning
* [scikit-image](https://scikit-image.org/) - Image processing
* [PyProj](https://pyproj4.github.io/pyproj/) - Coordinate transformations
* [GDAL](https://gdal.org/) - Geospatial data abstraction (optional)

## 👨‍💻 Author

**Florent Poux** - [3D Geodata Academy](https://learngeodata.eu)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
