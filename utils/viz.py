import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Global visualization state
_fig_3d = None
_ax_3d = None
_fig_2d = None
_ax_2d = None

def init_viz_3d(figsize=(12, 10)):
    global _fig_3d, _ax_3d
    _fig_3d = plt.figure(figsize=figsize)
    _ax_3d = _fig_3d.add_subplot(111, projection='3d')
    return _fig_3d, _ax_3d

def init_viz_2d(figsize=(12, 8)):
    global _fig_2d, _ax_2d
    _fig_2d, _ax_2d = plt.subplots(figsize=figsize)
    return _fig_2d, _ax_2d

def get_viz_3d():
    global _fig_3d, _ax_3d
    if _fig_3d is None:
        init_viz_3d()
    return _fig_3d, _ax_3d

def get_viz_2d():
    global _fig_2d, _ax_2d
    if _fig_2d is None:
        init_viz_2d()
    return _fig_2d, _ax_2d

def plot_points_3d(points, colors=None, labels=None, s=1, alpha=0.6, clear=True):
    fig, ax = get_viz_3d()
    if clear:
        ax.clear()

    if colors is None:
        colors = points[:, 2]

    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                      c=colors[mask] if colors.ndim > 1 else label,
                      s=s, alpha=alpha, label=f'Class {label}')
        ax.legend()
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=colors, s=s, alpha=alpha, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig, ax

def plot_points_2d(points, colors=None, s=1, alpha=0.6, clear=True):
    fig, ax = get_viz_2d()
    if clear:
        ax.clear()

    if colors is None:
        colors = 'b'

    ax.scatter(points[:, 0], points[:, 1], c=colors, s=s, alpha=alpha, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axis('equal')
    return fig, ax

def plot_heatmap_2d(x, y, values, resolution=100, clear=True):
    fig, ax = get_viz_2d()
    if clear:
        ax.clear()

    from scipy.interpolate import griddata
    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), values, (xi, yi), method='cubic')

    im = ax.imshow(zi, extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower', cmap='jet', aspect='auto')
    plt.colorbar(im, ax=ax)
    return fig, ax

def show_viz():
    plt.show()

def save_viz(filepath):
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
