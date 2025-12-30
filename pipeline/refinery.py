import numpy as np

def process_point_cloud_pipeline(points, colors=None, config=None):
    """Complete point cloud processing pipeline"""
    if config is None:
        config = get_default_config()

    results = {
        'original_points': points,
        'original_colors': colors
    }

    print(f"Input: {len(points)} points")

    if config['filter_outliers']:
        print("Filtering outliers...")
        from filters.sor import statistical_outlier_removal
        points, inlier_mask = statistical_outlier_removal(
            points,
            k=config['sor_k'],
            std_multiplier=config['sor_std']
        )

        if colors is not None:
            colors = colors[inlier_mask]

        results['filtered_points'] = points
        results['outlier_mask'] = inlier_mask
        print(f"After filtering: {len(points)} points")

    if config['segment_ground']:
        print("Segmenting ground...")
        from filters.ground_seg import extract_ground_points
        ground_points, non_ground_points, ground_mask = extract_ground_points(
            points,
            method=config['ground_method']
        )

        results['ground_points'] = ground_points
        results['non_ground_points'] = non_ground_points
        results['ground_mask'] = ground_mask
        print(f"Ground: {len(ground_points)}, Non-ground: {len(non_ground_points)}")

    if config['estimate_normals']:
        print("Estimating normals...")
        from features.normals import estimate_normals
        normals = estimate_normals(points, k=config['normal_k'])

        results['normals'] = normals

    if config['compute_features']:
        print("Computing geometric features...")
        from features.geometric import compute_covariance_features
        geometric_features = compute_covariance_features(points, k=config['feature_k'])

        results['geometric_features'] = geometric_features

    if config['segment_ground'] and config['compute_height']:
        print("Computing height features...")
        from features.height import compute_height_above_ground
        heights = compute_height_above_ground(points, ground_mask)

        results['heights'] = heights

    if colors is not None and config['compute_color_features']:
        print("Computing color features...")
        from features.color import rgb_to_hsv, compute_vegetation_index

        hsv = rgb_to_hsv(colors)
        veg_index = compute_vegetation_index(colors)

        results['hsv'] = hsv
        results['vegetation_index'] = veg_index

    if config['downsample']:
        print("Downsampling...")
        from filters.sampler import grid_sampling

        sampled_points, sampled_colors, inverse_indices = grid_sampling(
            points,
            config['downsample_voxel_size'],
            method='centroid',
            colors=colors
        )

        results['sampled_points'] = sampled_points
        results['sampled_colors'] = sampled_colors
        results['inverse_indices'] = inverse_indices
        print(f"Downsampled to: {len(sampled_points)} points")

        points_for_segmentation = sampled_points
    else:
        points_for_segmentation = points

    if config['segment']:
        print("Segmenting...")

        if config['segment_method'] == 'dbscan':
            from segmentation.dbscan import dbscan_clustering
            labels = dbscan_clustering(
                points_for_segmentation,
                eps=config['dbscan_eps'],
                min_samples=config['dbscan_min_samples']
            )

        elif config['segment_method'] == 'region_growing':
            from segmentation.region_grow import region_growing_normals

            if 'normals' in results:
                seg_normals = results['normals']
                if config['downsample']:
                    from features.normals import estimate_normals
                    seg_normals = estimate_normals(points_for_segmentation, k=20)
            else:
                from features.normals import estimate_normals
                seg_normals = estimate_normals(points_for_segmentation, k=20)

            labels = region_growing_normals(
                points_for_segmentation,
                seg_normals,
                k=20,
                angle_threshold=config['rg_angle_threshold']
            )

        elif config['segment_method'] == 'planes':
            from segmentation.ransac_shapes import detect_all_planes
            planes = detect_all_planes(
                points_for_segmentation,
                distance_threshold=config['plane_threshold'],
                max_planes=config['max_planes']
            )

            labels = np.full(len(points_for_segmentation), -1, dtype=np.int32)
            for i, plane_data in enumerate(planes):
                labels[plane_data['inliers']] = i

        results['segment_labels'] = labels

        unique_labels = np.unique(labels[labels >= 0])
        print(f"Found {len(unique_labels)} segments")

        if config['downsample']:
            print("Projecting labels to full resolution...")
            from filters.sampler import project_predictions_to_full_resolution
            full_labels = project_predictions_to_full_resolution(
                sampled_points,
                labels.astype(np.float32),
                points,
                k=5
            )
            full_labels = np.round(full_labels).astype(np.int32)
            results['full_labels'] = full_labels

    print("Pipeline complete!")

    return results

def export_results(results, output_path, config=None):
    """Export processed results to file"""
    if config is None:
        config = get_default_config()

    points = results.get('filtered_points', results['original_points'])
    colors = results.get('sampled_colors', results.get('original_colors'))
    normals = results.get('normals')

    scalar_fields = {}

    if 'full_labels' in results:
        scalar_fields['labels'] = results['full_labels'].astype(np.float32)
    elif 'segment_labels' in results:
        scalar_fields['labels'] = results['segment_labels'].astype(np.float32)

    if 'heights' in results:
        scalar_fields['height'] = results['heights']

    if 'vegetation_index' in results:
        scalar_fields['vegetation'] = results['vegetation_index']

    if 'geometric_features' in results:
        features = results['geometric_features']
        scalar_fields['linearity'] = features[:, 0]
        scalar_fields['planarity'] = features[:, 1]
        scalar_fields['sphericity'] = features[:, 2]

    if output_path.endswith('.ply'):
        from data.io_manager import write_ply_binary
        write_ply_binary(
            output_path,
            points,
            colors=colors,
            normals=normals,
            scalar_fields=scalar_fields
        )

    elif output_path.endswith('.las'):
        from data.io_manager import write_las_with_predictions
        predictions = scalar_fields.get('labels', np.zeros(len(points)))
        write_las_with_predictions(
            output_path,
            points,
            predictions.astype(np.uint8),
            colors=colors
        )

    elif output_path.endswith('.npz'):
        save_dict = {
            'points': points,
            'colors': colors,
            'normals': normals,
            **scalar_fields
        }
        save_dict = {k: v for k, v in save_dict.items() if v is not None}
        np.savez_compressed(output_path, **save_dict)

    print(f"Results exported to: {output_path}")

def get_default_config():
    """Get default pipeline configuration"""
    return {
        'filter_outliers': True,
        'sor_k': 20,
        'sor_std': 2.0,

        'segment_ground': True,
        'ground_method': 'csf',

        'estimate_normals': True,
        'normal_k': 20,

        'compute_features': True,
        'feature_k': 20,

        'compute_height': True,

        'compute_color_features': True,

        'downsample': True,
        'downsample_voxel_size': 0.1,

        'segment': True,
        'segment_method': 'dbscan',

        'dbscan_eps': 0.5,
        'dbscan_min_samples': 10,

        'rg_angle_threshold': 15.0,

        'plane_threshold': 0.01,
        'max_planes': 10
    }

def process_large_file(input_path, output_path, config=None, tile_size=50.0):
    """Process large files using tiling"""
    from data.io_manager import read_las_full, get_file_bounds
    from data.tiler import tile_point_cloud, save_tiles, load_tile, merge_tiles
    import os
    import tempfile

    if config is None:
        config = get_default_config()

    print(f"Processing large file: {input_path}")

    bounds_min, bounds_max = get_file_bounds(input_path)
    print(f"Bounds: {bounds_min} to {bounds_max}")

    points, colors, intensity, classification = read_las_full(input_path)

    print(f"Tiling point cloud...")
    tiles = tile_point_cloud(points, tile_size, overlap=5.0, colors=colors)

    print(f"Created {len(tiles)} tiles")

    temp_dir = tempfile.mkdtemp()

    processed_tiles = []

    for i, tile in enumerate(tiles):
        print(f"Processing tile {i+1}/{len(tiles)}...")

        tile_points = tile['points']
        tile_colors = tile.get('colors')

        results = process_point_cloud_pipeline(tile_points, tile_colors, config)

        processed_tile = {
            'id': tile['id'],
            'points': results.get('filtered_points', tile_points),
            'colors': tile_colors,
            'bounds_min': tile['bounds_min'],
            'bounds_max': tile['bounds_max']
        }

        if 'full_labels' in results:
            processed_tile['labels'] = results['full_labels']
        elif 'segment_labels' in results:
            processed_tile['labels'] = results['segment_labels']

        if 'normals' in results:
            processed_tile['normals'] = results['normals']

        processed_tiles.append(processed_tile)

    print("Merging tiles...")
    all_points = []
    all_colors = []
    all_labels = []

    for tile in processed_tiles:
        all_points.append(tile['points'])
        if tile.get('colors') is not None:
            all_colors.append(tile['colors'])
        if tile.get('labels') is not None:
            all_labels.append(tile['labels'])

    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors) if all_colors else None
    merged_labels = np.hstack(all_labels) if all_labels else None

    print(f"Merged: {len(merged_points)} points")

    scalar_fields = {}
    if merged_labels is not None:
        scalar_fields['labels'] = merged_labels.astype(np.float32)

    if output_path.endswith('.ply'):
        from data.io_manager import write_ply_binary
        write_ply_binary(output_path, merged_points, colors=merged_colors, scalar_fields=scalar_fields)
    else:
        from data.io_manager import write_las_with_predictions
        write_las_with_predictions(output_path, merged_points, merged_labels, colors=merged_colors)

    print(f"Results saved to: {output_path}")

    os.rmdir(temp_dir)
