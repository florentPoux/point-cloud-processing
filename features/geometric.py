import numpy as np
from scipy.spatial import cKDTree

def compute_covariance_features(points, k=20):
    """Compute eigenvalue-based geometric features"""
    tree = cKDTree(points)

    _, indices = tree.query(points, k=k)

    n_points = len(points)
    features = np.zeros((n_points, 7))

    for i in range(n_points):
        neighborhood = points[indices[i]]

        centroid = neighborhood.mean(axis=0)
        centered = neighborhood - centroid

        cov = np.dot(centered.T, centered) / k

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues, 1e-10)

        e1, e2, e3 = eigenvalues

        linearity = (e1 - e2) / e1
        planarity = (e2 - e3) / e1
        sphericity = e3 / e1
        omnivariance = (e1 * e2 * e3) ** (1/3)
        anisotropy = (e1 - e3) / e1
        eigenentropy = -((e1 * np.log(e1)) + (e2 * np.log(e2)) + (e3 * np.log(e3)))
        sum_eigenvalues = e1 + e2 + e3

        features[i] = [
            linearity,
            planarity,
            sphericity,
            omnivariance,
            anisotropy,
            eigenentropy,
            sum_eigenvalues
        ]

    return features

def compute_local_dimensionality(points, k=20):
    """Compute local dimensionality (1D, 2D, or 3D structure)"""
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)

    dimensionality = np.zeros(len(points))

    for i in range(len(points)):
        neighborhood = points[indices[i]]
        centroid = neighborhood.mean(axis=0)
        centered = neighborhood - centroid

        cov = np.dot(centered.T, centered) / k
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues, 1e-10)

        e1, e2, e3 = eigenvalues

        linearity = (e1 - e2) / e1
        planarity = (e2 - e3) / e1

        if linearity > planarity:
            dimensionality[i] = 1
        elif planarity > linearity:
            dimensionality[i] = 2
        else:
            dimensionality[i] = 3

    return dimensionality

def compute_curvature(points, k=20):
    """Compute local surface curvature"""
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)

    curvature = np.zeros(len(points))

    for i in range(len(points)):
        neighborhood = points[indices[i]]
        centroid = neighborhood.mean(axis=0)
        centered = neighborhood - centroid

        cov = np.dot(centered.T, centered) / k
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]

        curvature[i] = eigenvalues[2] / (eigenvalues.sum() + 1e-10)

    return curvature

def compute_local_point_density(points, k=20):
    """Compute local point density"""
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k)

    k_distance = distances[:, -1]

    volume = (4.0 / 3.0) * np.pi * (k_distance ** 3)
    density = k / volume

    return density

def compute_verticality(points, normals):
    """Compute verticality (alignment with Z axis)"""
    vertical_vector = np.array([0, 0, 1])

    dot_products = np.abs(np.dot(normals, vertical_vector))

    verticality = 1.0 - dot_products

    return verticality

def compute_moment_features(points, k=20):
    """Compute moment-based features"""
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)

    moments = np.zeros((len(points), 3))

    for i in range(len(points)):
        neighborhood = points[indices[i]]
        centroid = neighborhood.mean(axis=0)

        m1 = np.abs(neighborhood - centroid).mean(axis=0)
        m2 = ((neighborhood - centroid) ** 2).mean(axis=0)
        m3 = np.abs((neighborhood - centroid) ** 3).mean(axis=0)

        moments[i] = [m1.mean(), m2.mean(), m3.mean()]

    return moments

def compute_surface_variation(points, k=20):
    """Compute local surface variation"""
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)

    surface_variation = np.zeros(len(points))

    for i in range(len(points)):
        neighborhood = points[indices[i]]
        centroid = neighborhood.mean(axis=0)
        centered = neighborhood - centroid

        cov = np.dot(centered.T, centered) / k
        eigenvalues = np.linalg.eigvalsh(cov)

        surface_variation[i] = eigenvalues.min() / (eigenvalues.sum() + 1e-10)

    return surface_variation
