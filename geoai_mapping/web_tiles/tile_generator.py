import numpy as np
from pathlib import Path
import math

def latlon_to_tile(lat, lon, zoom):
    """Convert lat/lon to tile coordinates at given zoom level"""
    n = 2 ** zoom

    x_tile = int((lon + 180) / 360 * n)

    lat_rad = math.radians(lat)
    y_tile = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)

    return x_tile, y_tile

def tile_to_latlon(x_tile, y_tile, zoom):
    """Convert tile coordinates to lat/lon bounds"""
    n = 2 ** zoom

    lon_left = x_tile / n * 360 - 180

    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_tile / n)))
    lat_top = math.degrees(lat_rad)

    lon_right = (x_tile + 1) / n * 360 - 180

    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y_tile + 1) / n)))
    lat_bottom = math.degrees(lat_rad)

    return {
        'lat_min': lat_bottom,
        'lat_max': lat_top,
        'lon_min': lon_left,
        'lon_max': lon_right
    }

def generate_tiles_from_image(image, geotransform, output_dir, zoom_levels=[12, 13, 14], tile_size=256, epsg=4326):
    """Generate XYZ tiles from georeferenced image"""
    from geoai_mapping.core.coordinates import transform_coordinates

    output_path = Path(output_dir)

    height, width = image.shape[:2]

    corners = np.array([
        [geotransform[0], geotransform[3]],
        [geotransform[0] + width * geotransform[1], geotransform[3]],
        [geotransform[0], geotransform[3] + height * geotransform[5]],
        [geotransform[0] + width * geotransform[1], geotransform[3] + height * geotransform[5]]
    ])

    if epsg != 4326:
        corners_padded = np.column_stack([corners, np.zeros(4)])
        corners_wgs84 = transform_coordinates(corners_padded, epsg, 4326)
        corners = corners_wgs84[:, :2]

    lat_min, lat_max = corners[:, 1].min(), corners[:, 1].max()
    lon_min, lon_max = corners[:, 0].min(), corners[:, 0].max()

    tiles_generated = 0

    for zoom in zoom_levels:
        x_min, y_max = latlon_to_tile(lat_max, lon_min, zoom)
        x_max, y_min = latlon_to_tile(lat_min, lon_max, zoom)

        zoom_dir = output_path / str(zoom)
        zoom_dir.mkdir(parents=True, exist_ok=True)

        for x_tile in range(x_min, x_max + 1):
            x_dir = zoom_dir / str(x_tile)
            x_dir.mkdir(exist_ok=True)

            for y_tile in range(y_min, y_max + 1):
                tile_bounds = tile_to_latlon(x_tile, y_tile, zoom)

                tile_image = extract_tile_from_image(
                    image, geotransform,
                    tile_bounds['lon_min'], tile_bounds['lat_min'],
                    tile_bounds['lon_max'], tile_bounds['lat_max'],
                    tile_size, epsg
                )

                if tile_image is not None:
                    tile_path = x_dir / f"{y_tile}.png"
                    save_tile(tile_image, tile_path)
                    tiles_generated += 1

    return tiles_generated

def extract_tile_from_image(image, geotransform, lon_min, lat_min, lon_max, lat_max, tile_size, epsg):
    """Extract tile region from georeferenced image"""
    from geoai_mapping.core.georef import world_to_pixel

    if epsg != 4326:
        from geoai_mapping.core.coordinates import transform_coordinates

        corners_wgs84 = np.array([
            [lon_min, lat_min, 0],
            [lon_max, lat_max, 0]
        ])

        corners_proj = transform_coordinates(corners_wgs84, 4326, epsg)

        lon_min, lat_min = corners_proj[0, :2]
        lon_max, lat_max = corners_proj[1, :2]

    top_left = np.array([[lon_min, lat_max]])
    bottom_right = np.array([[lon_max, lat_min]])

    tl_pixel = world_to_pixel(top_left, geotransform)[0]
    br_pixel = world_to_pixel(bottom_right, geotransform)[0]

    x_min, y_min = tl_pixel.astype(int)
    x_max, y_max = br_pixel.astype(int)

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)
    y_max = min(image.shape[0], y_max)

    if x_max <= x_min or y_max <= y_min:
        return None

    crop = image[y_min:y_max, x_min:x_max]

    if crop.size == 0:
        return None

    from PIL import Image

    if len(crop.shape) == 2:
        pil_image = Image.fromarray(crop)
    else:
        pil_image = Image.fromarray(crop)

    tile = pil_image.resize((tile_size, tile_size), Image.Resampling.LANCZOS)

    return np.array(tile)

def save_tile(tile_image, filepath):
    """Save tile as PNG"""
    from PIL import Image

    img = Image.fromarray(tile_image)
    img.save(filepath, 'PNG', optimize=True)

def generate_tile_metadata(output_dir, zoom_levels, bounds):
    """Generate metadata.json for tile set"""
    import json

    metadata = {
        'name': 'GeoAI Map Tiles',
        'format': 'png',
        'minzoom': min(zoom_levels),
        'maxzoom': max(zoom_levels),
        'bounds': [
            bounds['lon_min'],
            bounds['lat_min'],
            bounds['lon_max'],
            bounds['lat_max']
        ],
        'center': [
            (bounds['lon_min'] + bounds['lon_max']) / 2,
            (bounds['lat_min'] + bounds['lat_max']) / 2,
            (min(zoom_levels) + max(zoom_levels)) // 2
        ]
    }

    with open(Path(output_dir) / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

def generate_leaflet_viewer(output_dir, metadata):
    """Generate HTML viewer using Leaflet"""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{metadata.get('name', 'Map Viewer')}</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; }}
        #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        var center = {metadata.get('center', [0, 0, 10])};
        var map = L.map('map').setView([center[1], center[0]], center[2]);

        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap contributors',
            maxZoom: 18
        }}).addTo(map);

        var customTiles = L.tileLayer('./{{z}}/{{x}}/{{y}}.png', {{
            minZoom: {metadata.get('minzoom', 0)},
            maxZoom: {metadata.get('maxzoom', 18)},
            tms: false,
            attribution: 'Custom data'
        }}).addTo(map);

        var bounds = {metadata.get('bounds', [-180, -90, 180, 90])};
        var rectangle = L.rectangle([
            [bounds[1], bounds[0]],
            [bounds[3], bounds[2]]
        ], {{color: "#ff7800", weight: 2, fillOpacity: 0}}).addTo(map);
    </script>
</body>
</html>"""

    with open(Path(output_dir) / 'viewer.html', 'w') as f:
        f.write(html)

def generate_pyramid_tiles(image, geotransform, output_dir, max_zoom=14, min_zoom=10):
    """Generate multi-resolution pyramid of tiles"""
    zoom_levels = list(range(min_zoom, max_zoom + 1))

    tiles_generated = generate_tiles_from_image(
        image, geotransform, output_dir, zoom_levels
    )

    return tiles_generated

def create_mbtiles(tile_dir, output_file, metadata=None):
    """Package tiles into MBTiles format (SQLite)"""
    import sqlite3
    import os

    conn = sqlite3.connect(output_file)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            name TEXT,
            value TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tiles (
            zoom_level INTEGER,
            tile_column INTEGER,
            tile_row INTEGER,
            tile_data BLOB
        )
    ''')

    if metadata:
        for key, value in metadata.items():
            cursor.execute('INSERT INTO metadata VALUES (?, ?)', (key, str(value)))

    tile_path = Path(tile_dir)

    for zoom_dir in tile_path.iterdir():
        if not zoom_dir.is_dir():
            continue

        zoom = int(zoom_dir.name)

        for x_dir in zoom_dir.iterdir():
            if not x_dir.is_dir():
                continue

            x = int(x_dir.name)

            for tile_file in x_dir.glob('*.png'):
                y = int(tile_file.stem)

                with open(tile_file, 'rb') as f:
                    tile_data = f.read()

                cursor.execute(
                    'INSERT INTO tiles VALUES (?, ?, ?, ?)',
                    (zoom, x, y, tile_data)
                )

    conn.commit()
    conn.close()
