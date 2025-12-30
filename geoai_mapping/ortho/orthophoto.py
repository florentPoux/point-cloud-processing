import numpy as np
from scipy.spatial import cKDTree

def generate_orthophoto(points, colors, resolution, method='nearest'):
    """Generate orthophoto from colored point cloud"""
    min_coords = np.min(points[:, :2], axis=0)
    max_coords = np.max(points[:, :2], axis=0)

    grid_width = int(np.ceil((max_coords[0] - min_coords[0]) / resolution))
    grid_height = int(np.ceil((max_coords[1] - min_coords[1]) / resolution))

    ortho = np.zeros((grid_height, grid_width, 3))
    counts = np.zeros((grid_height, grid_width))

    if method == 'nearest':
        for i, point in enumerate(points):
            px = int((point[0] - min_coords[0]) / resolution)
            py = int((point[1] - min_coords[1]) / resolution)

            if 0 <= px < grid_width and 0 <= py < grid_height:
                ortho[py, px] = colors[i]
                counts[py, px] += 1

    elif method == 'max_z':
        z_buffer = np.full((grid_height, grid_width), -np.inf)

        for i, point in enumerate(points):
            px = int((point[0] - min_coords[0]) / resolution)
            py = int((point[1] - min_coords[1]) / resolution)

            if 0 <= px < grid_width and 0 <= py < grid_height:
                if point[2] > z_buffer[py, px]:
                    z_buffer[py, px] = point[2]
                    ortho[py, px] = colors[i]
                    counts[py, px] = 1

    elif method == 'average':
        for i, point in enumerate(points):
            px = int((point[0] - min_coords[0]) / resolution)
            py = int((point[1] - min_coords[1]) / resolution)

            if 0 <= px < grid_width and 0 <= py < grid_height:
                ortho[py, px] += colors[i]
                counts[py, px] += 1

        valid_mask = counts > 0
        ortho[valid_mask] /= counts[valid_mask, np.newaxis]

    ortho = np.clip(ortho * 255, 0, 255).astype(np.uint8)

    origin = min_coords
    geotransform = [origin[0], resolution, 0, origin[1] + grid_height * resolution, 0, -resolution]

    return ortho, geotransform

def mosaic_orthophotos(orthos, geotransforms, blend_width=10):
    """Mosaic multiple orthophotos with feather blending"""
    all_bounds = []

    for geotrans in geotransforms:
        height, width = orthos[geotransforms.index(geotrans)].shape[:2]

        min_x = geotrans[0]
        max_y = geotrans[3]
        max_x = min_x + width * geotrans[1]
        min_y = max_y + height * geotrans[5]

        all_bounds.append([min_x, min_y, max_x, max_y])

    all_bounds = np.array(all_bounds)

    global_min_x = np.min(all_bounds[:, 0])
    global_min_y = np.min(all_bounds[:, 1])
    global_max_x = np.max(all_bounds[:, 2])
    global_max_y = np.max(all_bounds[:, 3])

    resolution = geotransforms[0][1]

    mosaic_width = int(np.ceil((global_max_x - global_min_x) / resolution))
    mosaic_height = int(np.ceil((global_max_y - global_min_y) / resolution))

    mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=float)
    weights = np.zeros((mosaic_height, mosaic_width), dtype=float)

    for ortho, geotrans in zip(orthos, geotransforms):
        height, width = ortho.shape[:2]

        offset_x = int((geotrans[0] - global_min_x) / resolution)
        offset_y = int((geotrans[3] - global_max_y) / abs(geotrans[5]))

        weight_map = create_blend_weights(height, width, blend_width)

        y_start = max(0, offset_y)
        y_end = min(mosaic_height, offset_y + height)
        x_start = max(0, offset_x)
        x_end = min(mosaic_width, offset_x + width)

        ortho_y_start = max(0, -offset_y)
        ortho_y_end = ortho_y_start + (y_end - y_start)
        ortho_x_start = max(0, -offset_x)
        ortho_x_end = ortho_x_start + (x_end - x_start)

        ortho_crop = ortho[ortho_y_start:ortho_y_end, ortho_x_start:ortho_x_end].astype(float)
        weight_crop = weight_map[ortho_y_start:ortho_y_end, ortho_x_start:ortho_x_end]

        mosaic[y_start:y_end, x_start:x_end] += ortho_crop * weight_crop[:, :, np.newaxis]
        weights[y_start:y_end, x_start:x_end] += weight_crop

    valid_mask = weights > 0
    mosaic[valid_mask] /= weights[valid_mask, np.newaxis]

    mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)

    mosaic_geotransform = [global_min_x, resolution, 0, global_max_y, 0, -resolution]

    return mosaic, mosaic_geotransform

def create_blend_weights(height, width, blend_width):
    """Create feather blend weight map"""
    weights = np.ones((height, width), dtype=float)

    for i in range(blend_width):
        alpha = i / blend_width

        weights[i, :] *= alpha
        weights[height - 1 - i, :] *= alpha
        weights[:, i] *= alpha
        weights[:, width - 1 - i] *= alpha

    return weights

def color_balance_orthophoto(ortho, target_mean=None):
    """Apply color balancing to orthophoto"""
    ortho_float = ortho.astype(float)

    if target_mean is None:
        target_mean = [128, 128, 128]

    balanced = ortho_float.copy()

    for channel in range(3):
        current_mean = np.mean(ortho_float[:, :, channel])

        if current_mean > 0:
            scale = target_mean[channel] / current_mean
            balanced[:, :, channel] = ortho_float[:, :, channel] * scale

    balanced = np.clip(balanced, 0, 255).astype(np.uint8)

    return balanced

def enhance_orthophoto(ortho, contrast=1.2, brightness=10):
    """Enhance orthophoto with contrast and brightness adjustment"""
    enhanced = ortho.astype(float)

    enhanced = enhanced * contrast + brightness

    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    return enhanced

def fill_holes_orthophoto(ortho, max_hole_size=5):
    """Fill small holes in orthophoto using inpainting"""
    from scipy.ndimage import binary_dilation, distance_transform_edt

    gray = np.mean(ortho, axis=2)
    mask = gray > 0

    holes = ~mask

    from skimage import morphology
    small_holes = morphology.remove_small_objects(holes, min_size=max_hole_size)

    filled = ortho.copy()

    for channel in range(3):
        distances, indices = distance_transform_edt(small_holes, return_indices=True)

        filled[:, :, channel][small_holes] = filled[:, :, channel][
            indices[0][small_holes], indices[1][small_holes]
        ]

    return filled

def export_geotiff(filepath, image, geotransform, epsg=4326):
    """Export orthophoto as GeoTIFF"""
    try:
        from osgeo import gdal, osr
        gdal.UseExceptions()

        driver = gdal.GetDriverByName('GTiff')

        if len(image.shape) == 2:
            bands = 1
            height, width = image.shape
        else:
            height, width, bands = image.shape

        dataset = driver.Create(
            filepath, width, height, bands, gdal.GDT_Byte,
            options=['COMPRESS=LZW', 'TILED=YES']
        )

        dataset.SetGeoTransform(geotransform)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        dataset.SetProjection(srs.ExportToWkt())

        if bands == 1:
            dataset.GetRasterBand(1).WriteArray(image)
        else:
            for band_idx in range(bands):
                dataset.GetRasterBand(band_idx + 1).WriteArray(image[:, :, band_idx])

        dataset.FlushCache()
        dataset = None

        return True

    except ImportError:
        return False
