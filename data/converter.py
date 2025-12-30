import numpy as np
import laspy

def las_to_ply(input_las, output_ply):
    """Convert LAS to PLY format"""
    from data.io_manager import read_las_full, write_ply_binary

    points, colors, intensity, classification = read_las_full(input_las)

    scalar_fields = {}
    if intensity is not None:
        scalar_fields['intensity'] = intensity
    if classification is not None:
        scalar_fields['classification'] = classification

    write_ply_binary(output_ply, points, colors=colors, scalar_fields=scalar_fields)

def ply_to_las(input_ply, output_las):
    """Convert PLY to LAS format"""
    points, colors = read_ply(input_ply)

    from data.io_manager import write_las_with_predictions
    predictions = np.zeros(len(points), dtype=np.uint8)
    write_las_with_predictions(output_las, points, predictions, colors=colors)

def read_ply(filepath):
    """Read PLY file"""
    with open(filepath, 'rb') as f:
        line = f.readline().decode('ascii').strip()
        if line != 'ply':
            raise ValueError("Not a PLY file")

        format_type = None
        n_vertices = 0
        properties = []

        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('format'):
                format_type = line.split()[1]
            elif line.startswith('element vertex'):
                n_vertices = int(line.split()[2])
            elif line.startswith('property'):
                parts = line.split()
                properties.append((parts[2], parts[1]))
            elif line == 'end_header':
                break

        if format_type == 'binary_little_endian':
            data = np.fromfile(f, dtype=np.uint8)
        else:
            raise NotImplementedError("Only binary PLY supported")

    points = np.zeros((n_vertices, 3))
    colors = None

    has_color = any(p[0] in ['red', 'green', 'blue'] for p in properties)
    if has_color:
        colors = np.zeros((n_vertices, 3))

    offset = 0
    for i in range(n_vertices):
        points[i, 0] = np.frombuffer(data[offset:offset+4], dtype=np.float32)[0]
        offset += 4
        points[i, 1] = np.frombuffer(data[offset:offset+4], dtype=np.float32)[0]
        offset += 4
        points[i, 2] = np.frombuffer(data[offset:offset+4], dtype=np.float32)[0]
        offset += 4

        if has_color:
            colors[i, 0] = data[offset] / 255.0
            offset += 1
            colors[i, 1] = data[offset] / 255.0
            offset += 1
            colors[i, 2] = data[offset] / 255.0
            offset += 1

    return points, colors

def npy_to_las(input_npy, output_las):
    """Convert NumPy array to LAS"""
    data = np.load(input_npy)

    if isinstance(data, np.ndarray):
        points = data
        colors = None
    else:
        points = data['points']
        colors = data.get('colors')

    from data.io_manager import write_las_with_predictions
    predictions = np.zeros(len(points), dtype=np.uint8)
    write_las_with_predictions(output_las, points, predictions, colors=colors)

def las_to_npy(input_las, output_npy):
    """Convert LAS to compressed NumPy"""
    from data.io_manager import read_las_full

    points, colors, intensity, classification = read_las_full(input_las)

    save_dict = {'points': points}
    if colors is not None:
        save_dict['colors'] = colors
    if intensity is not None:
        save_dict['intensity'] = intensity
    if classification is not None:
        save_dict['classification'] = classification

    np.savez_compressed(output_npy, **save_dict)

def xyz_to_las(input_xyz, output_las, has_intensity=False, has_rgb=False):
    """Convert ASCII XYZ to LAS"""
    if has_rgb and has_intensity:
        data = np.loadtxt(input_xyz)
        points = data[:, :3]
        intensity = data[:, 3]
        colors = data[:, 4:7] / 255.0
    elif has_rgb:
        data = np.loadtxt(input_xyz)
        points = data[:, :3]
        colors = data[:, 3:6] / 255.0
        intensity = None
    elif has_intensity:
        data = np.loadtxt(input_xyz)
        points = data[:, :3]
        intensity = data[:, 3]
        colors = None
    else:
        points = np.loadtxt(input_xyz)
        colors = None
        intensity = None

    from data.io_manager import write_las_with_predictions
    predictions = np.zeros(len(points), dtype=np.uint8)
    write_las_with_predictions(output_las, points, predictions, colors=colors, intensity=intensity)

def batch_convert(input_dir, output_dir, input_format, output_format):
    """Batch convert all files in directory"""
    import os
    import glob

    os.makedirs(output_dir, exist_ok=True)

    files = glob.glob(os.path.join(input_dir, f'*.{input_format}'))

    converters = {
        ('las', 'ply'): las_to_ply,
        ('ply', 'las'): ply_to_las,
        ('las', 'npy'): las_to_npy,
        ('npy', 'las'): npy_to_las,
    }

    converter_key = (input_format, output_format)
    if converter_key not in converters:
        raise ValueError(f"Conversion from {input_format} to {output_format} not supported")

    converter = converters[converter_key]

    for input_file in files:
        basename = os.path.basename(input_file)
        output_file = os.path.join(
            output_dir,
            basename.replace(f'.{input_format}', f'.{output_format}')
        )

        print(f"Converting {basename}...")
        converter(input_file, output_file)

    print(f"Converted {len(files)} files")
