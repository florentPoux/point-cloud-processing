#!/usr/bin/env python3

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.io_manager import read_las_full
from pipeline.refinery import process_point_cloud_pipeline, export_results, get_default_config, process_large_file
from utils.viz import init_viz_3d, plot_points_3d, show_viz

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Refinery - Process and segment point clouds')

    parser.add_argument('input', type=str, help='Input point cloud file (LAS/LAZ/PLY)')
    parser.add_argument('output', type=str, help='Output file path')

    parser.add_argument('--no-filter', action='store_true', help='Skip outlier filtering')
    parser.add_argument('--no-ground', action='store_true', help='Skip ground segmentation')
    parser.add_argument('--no-normals', action='store_true', help='Skip normal estimation')
    parser.add_argument('--no-segment', action='store_true', help='Skip segmentation')

    parser.add_argument('--segment-method', type=str, default='dbscan',
                       choices=['dbscan', 'region_growing', 'planes'],
                       help='Segmentation method')

    parser.add_argument('--downsample', type=float, default=0.1,
                       help='Voxel size for downsampling (0 to disable)')

    parser.add_argument('--dbscan-eps', type=float, default=0.5,
                       help='DBSCAN epsilon parameter')
    parser.add_argument('--dbscan-min-samples', type=int, default=10,
                       help='DBSCAN minimum samples')

    parser.add_argument('--visualize', action='store_true',
                       help='Visualize results')

    parser.add_argument('--large-file', action='store_true',
                       help='Use tiled processing for large files')
    parser.add_argument('--tile-size', type=float, default=50.0,
                       help='Tile size for large file processing')

    args = parser.parse_args()

    config = get_default_config()

    config['filter_outliers'] = not args.no_filter
    config['segment_ground'] = not args.no_ground
    config['estimate_normals'] = not args.no_normals
    config['segment'] = not args.no_segment
    config['segment_method'] = args.segment_method
    config['dbscan_eps'] = args.dbscan_eps
    config['dbscan_min_samples'] = args.dbscan_min_samples

    if args.downsample > 0:
        config['downsample'] = True
        config['downsample_voxel_size'] = args.downsample
    else:
        config['downsample'] = False

    print("=" * 60)
    print("POINT CLOUD REFINERY")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    try:
        if args.large_file:
            process_large_file(args.input, args.output, config, tile_size=args.tile_size)

        else:
            print("Loading point cloud...")
            points, colors, intensity, classification = read_las_full(args.input)

            print(f"Loaded {len(points)} points")

            results = process_point_cloud_pipeline(points, colors, config)

            export_results(results, args.output, config)

            if args.visualize:
                print("Visualizing results...")

                viz_points = results.get('filtered_points', points)

                if 'full_labels' in results:
                    labels = results['full_labels']
                elif 'segment_labels' in results:
                    labels = results['segment_labels']
                else:
                    labels = None

                init_viz_3d()
                plot_points_3d(viz_points, labels=labels, s=1)
                show_viz()

        print("=" * 60)
        print("PROCESSING COMPLETE!")
        print("=" * 60)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
