import numpy as np

def create_wall_geometry(wall_element, thickness=0.2, height=3.0):
    """Create wall geometry from detected plane"""
    from scan_to_bim.extraction.element_detector import extract_wall_boundaries

    boundaries = extract_wall_boundaries(wall_element['points'])

    if len(boundaries) == 0:
        return None

    main_boundary = boundaries[0]

    z_min = wall_element['points'][:, 2].min()
    z_max = z_min + height

    vertices = []
    faces = []

    for i, point_2d in enumerate(main_boundary):
        vertices.append([point_2d[0], point_2d[1], z_min])
        vertices.append([point_2d[0], point_2d[1], z_max])

    vertices = np.array(vertices)

    for i in range(0, len(vertices) - 2, 2):
        faces.append([i, i + 2, i + 3, i + 1])

    return {
        'vertices': vertices,
        'faces': faces,
        'thickness': thickness,
        'height': height,
        'boundary': main_boundary
    }

def create_floor_geometry(floor_element):
    """Create floor geometry"""
    points = floor_element['points']

    from scipy.spatial import Delaunay

    tri = Delaunay(points[:, :2])

    vertices = points
    faces = tri.simplices

    return {
        'vertices': vertices,
        'faces': faces,
        'area': calculate_polygon_area(points[:, :2])
    }

def calculate_polygon_area(points_2d):
    """Calculate 2D polygon area"""
    x = points_2d[:, 0]
    y = points_2d[:, 1]

    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    return area

def create_door_geometry(door_opening, wall_thickness=0.2):
    """Create door geometry"""
    width = door_opening['width']
    height = door_opening['height']
    center = door_opening['center']

    vertices = [
        [center[0] - width/2, center[1], center[2]],
        [center[0] + width/2, center[1], center[2]],
        [center[0] + width/2, center[1], center[2] + height],
        [center[0] - width/2, center[1], center[2] + height]
    ]

    faces = [[0, 1, 2, 3]]

    return {
        'vertices': np.array(vertices),
        'faces': faces,
        'width': width,
        'height': height,
        'type': 'door'
    }

def create_window_geometry(window_opening):
    """Create window geometry"""
    width = window_opening['width']
    height = window_opening['height']
    center = window_opening['center']

    vertices = [
        [center[0] - width/2, center[1], center[2] - height/2],
        [center[0] + width/2, center[1], center[2] - height/2],
        [center[0] + width/2, center[1], center[2] + height/2],
        [center[0] - width/2, center[1], center[2] + height/2]
    ]

    faces = [[0, 1, 2, 3]]

    return {
        'vertices': np.array(vertices),
        'faces': faces,
        'width': width,
        'height': height,
        'type': 'window'
    }

def create_column_geometry(column_element, subdivisions=16):
    """Create column geometry"""
    center = column_element['center']
    radius = column_element['radius']
    height = column_element['height']

    vertices = []
    faces = []

    for i in range(subdivisions):
        angle = 2 * np.pi * i / subdivisions

        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)

        vertices.append([x, y, center[2]])
        vertices.append([x, y, center[2] + height])

    vertices = np.array(vertices)

    for i in range(0, len(vertices) - 2, 2):
        next_i = (i + 2) % len(vertices)
        faces.append([i, next_i, next_i + 1, i + 1])

    return {
        'vertices': vertices,
        'faces': faces,
        'radius': radius,
        'height': height
    }

def generate_bim_model(building_elements):
    """Generate complete BIM model from elements"""
    bim_model = {
        'walls': [],
        'floors': [],
        'ceilings': [],
        'doors': [],
        'windows': [],
        'columns': [],
        'metadata': {}
    }

    for wall in building_elements['walls']:
        geometry = create_wall_geometry(wall)
        if geometry is not None:
            bim_model['walls'].append(geometry)

    for floor in building_elements['floors']:
        geometry = create_floor_geometry(floor)
        bim_model['floors'].append(geometry)

    for ceiling in building_elements['ceilings']:
        geometry = create_floor_geometry(ceiling)
        bim_model['ceilings'].append(geometry)

    for door in building_elements['doors']:
        geometry = create_door_geometry(door)
        bim_model['doors'].append(geometry)

    for window in building_elements['windows']:
        geometry = create_window_geometry(window)
        bim_model['windows'].append(geometry)

    for column in building_elements['columns']:
        geometry = create_column_geometry(column)
        bim_model['columns'].append(geometry)

    total_floor_area = sum(floor['area'] for floor in bim_model['floors'])

    bim_model['metadata'] = {
        'num_walls': len(bim_model['walls']),
        'num_floors': len(bim_model['floors']),
        'num_doors': len(bim_model['doors']),
        'num_windows': len(bim_model['windows']),
        'num_columns': len(bim_model['columns']),
        'total_floor_area': total_floor_area
    }

    return bim_model
