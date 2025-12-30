#!/usr/bin/env python3
"""
Zero-shot point cloud classification using CLIP
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.io_manager import read_las_full
from zero_shot_foundation.models.clip_3d import (
    load_clip_model,
    zero_shot_classify_point_cloud,
    encode_text_with_ensembling
)
from zero_shot_foundation.prompting.templates import create_classification_prompts

def main(input_file, class_names, domain='general'):
    """Zero-shot classification example"""
    print("=" * 60)
    print("ZERO-SHOT POINT CLOUD CLASSIFICATION")
    print("=" * 60)

    print(f"\n1. Loading point cloud: {input_file}")
    points, colors, _, _ = read_las_full(input_file)
    print(f"   Loaded {len(points):,} points")

    print(f"\n2. Loading CLIP model...")
    clip_model, preprocess = load_clip_model(model_name='ViT-B/32', device='cuda')

    if clip_model is None:
        print("   ERROR: CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
        return

    print(f"\n3. Creating prompts for classes: {', '.join(class_names)}")
    prompt_templates = create_classification_prompts(class_names, domain=domain)

    print(f"\n4. Encoding text prompts...")
    text_features = encode_text_with_ensembling(
        class_names, clip_model, device='cuda'
    )
    print(f"   Text features shape: {text_features.shape}")

    print(f"\n5. Running zero-shot classification...")
    predicted_class, confidence, similarities = zero_shot_classify_point_cloud(
        points, colors, class_names, clip_model, preprocess, num_views=8
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nPredicted class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")
    print("\nAll class similarities:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {similarities[i]:.4f}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Zero-shot point cloud classification')
    parser.add_argument('input', type=str, help='Input LAS file')
    parser.add_argument('--classes', type=str, nargs='+',
                       default=['tree', 'building', 'car', 'ground'],
                       help='Class names for classification')
    parser.add_argument('--domain', type=str, default='outdoor',
                       choices=['general', 'indoor', 'outdoor', 'building'],
                       help='Domain for prompt templates')

    args = parser.parse_args()

    main(args.input, args.classes, args.domain)
