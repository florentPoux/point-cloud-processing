import numpy as np
import json
from pathlib import Path

def create_thumbnail(positions, colors, width=256, height=256):
    """Generate thumbnail image for marketplace preview"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    if isinstance(positions, np.ndarray):
        pos = positions
        col = colors
    else:
        pos = positions.cpu().numpy()
        col = colors.cpu().numpy()

    sample_size = min(10000, len(pos))
    indices = np.random.choice(len(pos), sample_size, replace=False)

    ax.scatter(pos[indices, 0], pos[indices, 1], pos[indices, 2],
              c=col[indices], s=1, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    thumbnail_path = 'thumbnail.png'
    plt.savefig(thumbnail_path, dpi=100, bbox_inches='tight')
    plt.close()

    return thumbnail_path

def package_for_marketplace(model_path, metadata, output_dir):
    """Package Gaussian splat model for marketplace distribution"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    from gaussian_splatting.core.gaussian import load_gaussian_model

    positions, colors, scales, rotations, opacities, sh_coeffs = load_gaussian_model(model_path)

    thumbnail_path = create_thumbnail(positions, colors)

    from gaussian_splatting.streaming.serializer import save_streaming_format

    streaming_dir = output_path / 'streaming'
    manifest = save_streaming_format(positions, colors, scales, rotations, opacities, streaming_dir)

    package_metadata = {
        'title': metadata.get('title', 'Untitled Gaussian Splat'),
        'description': metadata.get('description', ''),
        'author': metadata.get('author', ''),
        'license': metadata.get('license', 'CC BY 4.0'),
        'tags': metadata.get('tags', []),
        'n_gaussians': len(positions),
        'bounding_box': {
            'min': positions.min(axis=0).tolist(),
            'max': positions.max(axis=0).tolist()
        },
        'file_size_mb': Path(model_path).stat().st_size / (1024 * 1024),
        'streaming_enabled': True,
        'thumbnail': 'thumbnail.png'
    }

    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(package_metadata, f, indent=2)

    readme = f"""# {package_metadata['title']}

{package_metadata['description']}

## Model Information

- **Gaussians**: {package_metadata['n_gaussians']:,}
- **File Size**: {package_metadata['file_size_mb']:.2f} MB
- **Author**: {package_metadata['author']}
- **License**: {package_metadata['license']}

## Usage

This model uses Gaussian Splatting for real-time rendering. Load it using the provided web viewer or compatible 3D software.

## Bounding Box

- Min: {package_metadata['bounding_box']['min']}
- Max: {package_metadata['bounding_box']['max']}
"""

    readme_path = output_path / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme)

    print(f"Package created at: {output_path}")
    print(f"  - Metadata: {metadata_path}")
    print(f"  - Thumbnail: {thumbnail_path}")
    print(f"  - Streaming data: {streaming_dir}")

    return package_metadata

def upload_to_marketplace(package_dir, api_key, marketplace_url='https://api.marketplace.example.com'):
    """Upload packaged model to online marketplace"""
    import requests

    package_path = Path(package_dir)

    with open(package_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    files = {
        'thumbnail': open(package_path / 'thumbnail.png', 'rb'),
        'metadata': json.dumps(metadata)
    }

    headers = {
        'Authorization': f'Bearer {api_key}'
    }

    response = requests.post(
        f'{marketplace_url}/upload',
        files=files,
        headers=headers
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Successfully uploaded! Asset ID: {result['asset_id']}")
        print(f"View at: {result['url']}")
        return result
    else:
        print(f"Upload failed: {response.status_code}")
        print(response.text)
        return None

def estimate_pricing(n_gaussians, complexity_score=1.0):
    """Estimate marketplace pricing based on model complexity"""
    base_price = 5.0

    if n_gaussians < 100000:
        price = base_price
    elif n_gaussians < 500000:
        price = base_price * 2
    elif n_gaussians < 1000000:
        price = base_price * 3
    else:
        price = base_price * 5

    price *= complexity_score

    return {
        'suggested_price_usd': round(price, 2),
        'complexity_tier': 'low' if n_gaussians < 100000 else 'medium' if n_gaussians < 500000 else 'high'
    }

def create_marketplace_listing(package_metadata, pricing_info):
    """Create marketplace listing description"""
    listing = {
        'title': package_metadata['title'],
        'description': package_metadata['description'],
        'price': pricing_info['suggested_price_usd'],
        'category': 'gaussian-splats',
        'technical_details': {
            'gaussians': package_metadata['n_gaussians'],
            'file_size_mb': package_metadata['file_size_mb'],
            'streaming': package_metadata['streaming_enabled']
        },
        'tags': package_metadata['tags'],
        'license': package_metadata['license'],
        'author': package_metadata['author']
    }

    return listing
