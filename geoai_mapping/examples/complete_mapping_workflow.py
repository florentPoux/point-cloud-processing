#!/usr/bin/env python3
"""
Complete GeoAI mapping workflow from point cloud to web tiles
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.io_manager import read_las_full
from filters.ground_seg import extract_ground_points
from geoai_mapping.terrain.dem_dsm import (
    generate_dsm, generate_dtm, compute_chm,
    compute_slope, compute_aspect, compute_hillshade
)
from geoai_mapping.ortho.orthophoto import generate_orthophoto, export_geotiff
from geoai_mapping.classification.land_cover import (
    extract_features_for_classification,
    create_land_cover_map
)
from geoai_mapping.web_tiles.tile_generator import (
    generate_tiles_from_image,
    generate_tile_metadata,
    generate_leaflet_viewer
)

def main(input_las, output_dir, resolution=0.5):
    """Complete mapping workflow"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GEOAI MAPPING WORKFLOW")
    print("=" * 60)

    print(f"\n1. Loading point cloud: {input_las}")
    points, colors, _, _ = read_las_full(input_las)
    print(f"   Loaded {len(points):,} points")

    print("\n2. Segmenting ground points...")
    ground_points, non_ground_points, ground_mask = extract_ground_points(
        points, method='csf'
    )
    print(f"   Ground: {len(ground_points):,} points")
    print(f"   Non-ground: {len(non_ground_points):,} points")

    print("\n3. Generating Digital Surface Model (DSM)...")
    dsm, geotransform = generate_dsm(points, resolution)
    print(f"   DSM shape: {dsm.shape}")

    print("\n4. Generating Digital Terrain Model (DTM)...")
    dtm, _ = generate_dtm(ground_points, resolution, smooth=True)
    print(f"   DTM shape: {dtm.shape}")

    print("\n5. Computing Canopy Height Model (CHM)...")
    chm = compute_chm(dsm, dtm)
    print(f"   Max vegetation height: {np.max(chm):.2f}m")

    print("\n6. Computing terrain derivatives...")
    slope = compute_slope(dtm, resolution)
    aspect = compute_aspect(dtm, resolution)
    hillshade = compute_hillshade(dtm, resolution)
    print(f"   Max slope: {np.max(slope):.2f}°")

    if colors is not None:
        print("\n7. Generating orthophoto...")
        ortho, ortho_geotrans = generate_orthophoto(
            points, colors, resolution, method='max_z'
        )
        print(f"   Orthophoto shape: {ortho.shape}")

        ortho_path = output_path / 'orthophoto.tif'
        export_geotiff(str(ortho_path), ortho, ortho_geotrans, epsg=32632)
        print(f"   Saved: {ortho_path}")

    print("\n8. Classifying land cover...")
    from features.normals import estimate_normals
    normals = estimate_normals(points, k=20)

    features = extract_features_for_classification(
        points, colors, normals, k_neighbors=20
    )

    vegetation_mask = chm[
        ((points[:, 1] - geotransform[3]) / geotransform[5]).astype(int),
        ((points[:, 0] - geotransform[0]) / geotransform[1]).astype(int)
    ] > 2.0

    labels = np.zeros(len(points), dtype=int)
    labels[ground_mask] = 0
    labels[vegetation_mask] = 1
    labels[~ground_mask & ~vegetation_mask] = 2

    land_cover_map, lc_geotrans = create_land_cover_map(
        points, labels, resolution, num_classes=3
    )
    print(f"   Land cover map shape: {land_cover_map.shape}")

    print("\n9. Exporting GeoTIFFs...")
    export_geotiff(str(output_path / 'dsm.tif'), dsm.astype(np.float32), geotransform, epsg=32632)
    export_geotiff(str(output_path / 'dtm.tif'), dtm.astype(np.float32), geotransform, epsg=32632)
    export_geotiff(str(output_path / 'chm.tif'), chm.astype(np.float32), geotransform, epsg=32632)
    export_geotiff(str(output_path / 'slope.tif'), slope.astype(np.float32), geotransform, epsg=32632)
    export_geotiff(str(output_path / 'hillshade.tif'), hillshade, geotransform, epsg=32632)
    export_geotiff(str(output_path / 'land_cover.tif'), land_cover_map.astype(np.uint8), lc_geotrans, epsg=32632)

    if colors is not None:
        print("\n10. Generating web map tiles...")
        tiles_dir = output_path / 'tiles'
        tiles_generated = generate_tiles_from_image(
            ortho, ortho_geotrans, tiles_dir,
            zoom_levels=[12, 13, 14], epsg=32632
        )
        print(f"    Generated {tiles_generated} tiles")

        bounds = {
            'lon_min': ortho_geotrans[0],
            'lat_min': ortho_geotrans[3] + ortho.shape[0] * ortho_geotrans[5],
            'lon_max': ortho_geotrans[0] + ortho.shape[1] * ortho_geotrans[1],
            'lat_max': ortho_geotrans[3]
        }

        generate_tile_metadata(tiles_dir, [12, 13, 14], bounds)
        generate_leaflet_viewer(tiles_dir, {
            'name': 'GeoAI Orthophoto',
            'minzoom': 12,
            'maxzoom': 14,
            'bounds': [bounds['lon_min'], bounds['lat_min'], bounds['lon_max'], bounds['lat_max']],
            'center': [(bounds['lon_min'] + bounds['lon_max']) / 2,
                      (bounds['lat_min'] + bounds['lat_max']) / 2, 13]
        })
        print(f"    Web viewer: {tiles_dir}/viewer.html")

    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE!")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_path}")
    print("  - dsm.tif: Digital Surface Model")
    print("  - dtm.tif: Digital Terrain Model")
    print("  - chm.tif: Canopy Height Model")
    print("  - slope.tif: Slope map")
    print("  - hillshade.tif: Hillshade visualization")
    print("  - land_cover.tif: Land cover classification")
    if colors is not None:
        print("  - orthophoto.tif: Orthophoto")
        print("  - tiles/: Web map tiles")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Complete GeoAI mapping workflow')
    parser.add_argument('input', type=str, help='Input LAS file')
    parser.add_argument('--output', type=str, default='./mapping_output', help='Output directory')
    parser.add_argument('--resolution', type=float, default=0.5, help='Output resolution in meters')

    args = parser.parse_args()

    main(args.input, args.output, args.resolution)
