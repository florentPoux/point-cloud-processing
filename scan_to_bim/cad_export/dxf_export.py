import numpy as np

def export_to_dxf(bim_model, filepath, layer_prefix='BUILDING'):
    """Export BIM model to DXF format"""
    try:
        import ezdxf
    except ImportError:
        return export_simple_dxf(bim_model, filepath)

    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    for i, wall in enumerate(bim_model['walls']):
        layer_name = f"{layer_prefix}_WALL_{i}"
        doc.layers.new(name=layer_name, dxfattribs={'color': 7})

        boundary = wall['boundary']

        for j in range(len(boundary)):
            start = boundary[j]
            end = boundary[(j + 1) % len(boundary)]

            msp.add_line(
                (start[0], start[1]),
                (end[0], end[1]),
                dxfattribs={'layer': layer_name}
            )

    for i, door in enumerate(bim_model['doors']):
        layer_name = f"{layer_prefix}_DOOR_{i}"
        doc.layers.new(name=layer_name, dxfattribs={'color': 2})

        vertices = door['vertices']

        for j in range(len(vertices)):
            start = vertices[j]
            end = vertices[(j + 1) % len(vertices)]

            msp.add_line(
                (start[0], start[1]),
                (end[0], end[1]),
                dxfattribs={'layer': layer_name}
            )

    for i, window in enumerate(bim_model['windows']):
        layer_name = f"{layer_prefix}_WINDOW_{i}"
        doc.layers.new(name=layer_name, dxfattribs={'color': 4})

        vertices = window['vertices']

        for j in range(len(vertices)):
            start = vertices[j]
            end = vertices[(j + 1) % len(vertices)]

            msp.add_line(
                (start[0], start[1]),
                (end[0], end[1]),
                dxfattribs={'layer': layer_name}
            )

    doc.saveas(filepath)

    return True

def export_simple_dxf(bim_model, filepath):
    """Export simple DXF without ezdxf library"""
    with open(filepath, 'w') as f:
        f.write("0\nSECTION\n2\nENTITIES\n")

        for wall in bim_model['walls']:
            boundary = wall['boundary']

            for j in range(len(boundary)):
                start = boundary[j]
                end = boundary[(j + 1) % len(boundary)]

                f.write(f"0\nLINE\n8\nWALL\n")
                f.write(f"10\n{start[0]}\n20\n{start[1]}\n30\n0.0\n")
                f.write(f"11\n{end[0]}\n21\n{end[1]}\n31\n0.0\n")

        f.write("0\nENDSEC\n0\nEOF\n")

    return True

def export_floor_plan(bim_model, filepath, scale=100):
    """Export 2D floor plan"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon

    fig, ax = plt.subplots(figsize=(12, 12))

    for wall in bim_model['walls']:
        boundary = wall['boundary']
        poly = Polygon(boundary, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(poly)

    for door in bim_model['doors']:
        vertices = door['vertices'][:, :2]
        rect = Rectangle(
            vertices[0], door['width'], door['height'],
            fill=False, edgecolor='blue', linewidth=1.5
        )
        ax.add_patch(rect)

    for window in bim_model['windows']:
        vertices = window['vertices'][:, :2]
        rect = Rectangle(
            vertices[0], window['width'], window['height'],
            fill=False, edgecolor='green', linewidth=1
        )
        ax.add_patch(rect)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Floor Plan')

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    return True
