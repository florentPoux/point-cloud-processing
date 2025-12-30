import numpy as np
import struct

def serialize_gaussian_binary(positions, colors, scales, rotations, opacities):
    """Serialize Gaussian data to compact binary format"""
    n_gaussians = len(positions)

    header = struct.pack('I', n_gaussians)

    binary_data = bytearray(header)

    for i in range(n_gaussians):
        pos = positions[i]
        color = colors[i]
        scale = scales[i]
        rot = rotations[i]
        opacity = opacities[i]

        binary_data.extend(struct.pack('fff', float(pos[0]), float(pos[1]), float(pos[2])))

        binary_data.extend(struct.pack('fff', float(color[0]), float(color[1]), float(color[2])))

        binary_data.extend(struct.pack('fff', float(scale[0]), float(scale[1]), float(scale[2])))

        binary_data.extend(struct.pack('ffff', float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])))

        binary_data.extend(struct.pack('f', float(opacity)))

    return bytes(binary_data)

def deserialize_gaussian_binary(binary_data):
    """Deserialize binary Gaussian data"""
    n_gaussians = struct.unpack('I', binary_data[:4])[0]

    offset = 4

    positions = []
    colors = []
    scales = []
    rotations = []
    opacities = []

    record_size = 3 + 3 + 3 + 4 + 1

    for i in range(n_gaussians):
        pos = struct.unpack('fff', binary_data[offset:offset+12])
        offset += 12

        color = struct.unpack('fff', binary_data[offset:offset+12])
        offset += 12

        scale = struct.unpack('fff', binary_data[offset:offset+12])
        offset += 12

        rot = struct.unpack('ffff', binary_data[offset:offset+16])
        offset += 16

        opacity = struct.unpack('f', binary_data[offset:offset+4])[0]
        offset += 4

        positions.append(pos)
        colors.append(color)
        scales.append(scale)
        rotations.append(rot)
        opacities.append(opacity)

    return {
        'positions': np.array(positions, dtype=np.float32),
        'colors': np.array(colors, dtype=np.float32),
        'scales': np.array(scales, dtype=np.float32),
        'rotations': np.array(rotations, dtype=np.float32),
        'opacities': np.array(opacities, dtype=np.float32)
    }

def create_streaming_chunks(positions, colors, scales, rotations, opacities, chunk_size=10000):
    """Create chunks for progressive streaming"""
    n_gaussians = len(positions)
    n_chunks = (n_gaussians + chunk_size - 1) // chunk_size

    chunks = []

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, n_gaussians)

        chunk_data = serialize_gaussian_binary(
            positions[start_idx:end_idx],
            colors[start_idx:end_idx],
            scales[start_idx:end_idx],
            rotations[start_idx:end_idx],
            opacities[start_idx:end_idx]
        )

        chunks.append({
            'chunk_id': i,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'data': chunk_data,
            'size': len(chunk_data)
        })

    return chunks

def compress_chunk(chunk_data):
    """Compress chunk data using zlib"""
    import zlib
    return zlib.compress(chunk_data, level=9)

def decompress_chunk(compressed_data):
    """Decompress chunk data"""
    import zlib
    return zlib.decompress(compressed_data)

def save_streaming_format(positions, colors, scales, rotations, opacities, output_dir, chunk_size=10000):
    """Save Gaussians in streaming-ready format"""
    import os
    import json

    os.makedirs(output_dir, exist_ok=True)

    chunks = create_streaming_chunks(positions, colors, scales, rotations, opacities, chunk_size)

    manifest = {
        'n_gaussians': len(positions),
        'n_chunks': len(chunks),
        'chunk_size': chunk_size,
        'chunks': []
    }

    for chunk in chunks:
        chunk_filename = f'chunk_{chunk["chunk_id"]:06d}.bin'
        chunk_path = os.path.join(output_dir, chunk_filename)

        compressed = compress_chunk(chunk['data'])

        with open(chunk_path, 'wb') as f:
            f.write(compressed)

        manifest['chunks'].append({
            'id': chunk['chunk_id'],
            'filename': chunk_filename,
            'start_idx': chunk['start_idx'],
            'end_idx': chunk['end_idx'],
            'compressed_size': len(compressed),
            'uncompressed_size': chunk['size']
        })

    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return manifest
