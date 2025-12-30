import numpy as np
from scipy.spatial import Delaunay
from scipy.ndimage import binary_erosion, binary_dilation

def generate_mesh_from_dem(dem, geotransform, simplification=1):
    """Generate triangle mesh from DEM"""
    height, width = dem.shape

    step = max(1, simplification)

    y_indices, x_indices = np.mgrid[0:height:step, 0:width:step]
    y_indices = y_indices.ravel()
    x_indices = x_indices.ravel()

    x_geo = geotransform[0] + x_indices * geotransform[1]
    y_geo = geotransform[3] + y_indices * geotransform[5]
    z_geo = dem[y_indices, x_indices]

    valid = ~np.isnan(z_geo)
    vertices = np.column_stack([x_geo[valid], y_geo[valid], z_geo[valid]])

    points_2d = vertices[:, :2]
    tri = Delaunay(points_2d)

    return vertices, tri.simplices

def generate_mesh_from_points(points, alpha=None):
    """Generate triangle mesh from point cloud using Delaunay"""
    tri = Delaunay(points[:, :2])

    if alpha is None:
        return points, tri.simplices

    filtered_simplices = []

    for simplex in tri.simplices:
        pts = points[simplex]

        edge_lengths = [
            np.linalg.norm(pts[1] - pts[0]),
            np.linalg.norm(pts[2] - pts[1]),
            np.linalg.norm(pts[0] - pts[2])
        ]

        if max(edge_lengths) < alpha:
            filtered_simplices.append(simplex)

    return points, np.array(filtered_simplices)

def compute_mesh_normals(vertices, faces):
    """Compute per-face normals for mesh"""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0

    normals = np.cross(edge1, edge2)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (lengths + 1e-10)

    return normals

def compute_vertex_normals(vertices, faces):
    """Compute smooth per-vertex normals"""
    vertex_normals = np.zeros_like(vertices)
    counts = np.zeros(len(vertices))

    for face in faces:
        v0, v1, v2 = vertices[face]

        edge1 = v1 - v0
        edge2 = v2 - v0

        normal = np.cross(edge1, edge2)

        for idx in face:
            vertex_normals[idx] += normal
            counts[idx] += 1

    counts = np.maximum(counts, 1)
    vertex_normals = vertex_normals / counts[:, np.newaxis]

    lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / (lengths + 1e-10)

    return vertex_normals

def simplify_mesh_quadric(vertices, faces, target_faces):
    """Simplify mesh using quadric error metrics"""
    num_remove = len(faces) - target_faces

    if num_remove <= 0:
        return vertices, faces

    edge_costs = []

    for i, face in enumerate(faces):
        for j in range(3):
            v1_idx = face[j]
            v2_idx = face[(j + 1) % 3]

            edge = (min(v1_idx, v2_idx), max(v1_idx, v2_idx))
            edge_length = np.linalg.norm(vertices[v1_idx] - vertices[v2_idx])

            edge_costs.append((edge_length, edge, i))

    edge_costs.sort()

    faces_to_remove = set()

    for cost, edge, face_idx in edge_costs[:num_remove]:
        faces_to_remove.add(face_idx)

    simplified_faces = [face for i, face in enumerate(faces) if i not in faces_to_remove]

    return vertices, np.array(simplified_faces)

def export_mesh_obj(filepath, vertices, faces, normals=None):
    """Export mesh to OBJ format"""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        if normals is not None:
            for n in normals:
                f.write(f"vn {n[0]} {n[1]} {n[2]}\n")

        for face in faces:
            if normals is not None:
                f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
            else:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def export_mesh_ply(filepath, vertices, faces, colors=None):
    """Export mesh to PLY format"""
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")

        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for i, v in enumerate(vertices):
            if colors is not None:
                c = (colors[i] * 255).astype(np.uint8)
                f.write(f"{v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")
            else:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

def texture_mesh_from_ortho(vertices, faces, ortho_image, ortho_geotransform):
    """Generate texture coordinates from orthophoto"""
    uv_coords = np.zeros((len(vertices), 2))

    img_height, img_width = ortho_image.shape[:2]

    for i, vertex in enumerate(vertices):
        x_geo, y_geo = vertex[0], vertex[1]

        px = (x_geo - ortho_geotransform[0]) / ortho_geotransform[1]
        py = (y_geo - ortho_geotransform[3]) / ortho_geotransform[5]

        u = px / img_width
        v = py / img_height

        uv_coords[i] = [u, v]

    return uv_coords

def extract_building_footprints(dsm, dtm, min_height=3.0, min_area=20.0, resolution=0.5):
    """Extract building footprints from DSM/DTM difference"""
    from skimage import measure, morphology

    ndsm = dsm - dtm

    building_mask = ndsm > min_height

    building_mask = morphology.remove_small_objects(building_mask, min_size=int(min_area / (resolution ** 2)))
    building_mask = morphology.remove_small_holes(building_mask, area_threshold=int(min_area / (resolution ** 2)))

    building_mask = binary_erosion(building_mask, iterations=1)
    building_mask = binary_dilation(building_mask, iterations=1)

    labeled = measure.label(building_mask)

    footprints = []

    for region in measure.regionprops(labeled):
        if region.area * (resolution ** 2) < min_area:
            continue

        contours = measure.find_contours(labeled == region.label, 0.5)

        if len(contours) > 0:
            contour = contours[0]

            simplified = contour[::max(1, len(contour) // 20)]

            footprints.append({
                'area': region.area * (resolution ** 2),
                'centroid': region.centroid,
                'bbox': region.bbox,
                'contour': simplified
            })

    return footprints
