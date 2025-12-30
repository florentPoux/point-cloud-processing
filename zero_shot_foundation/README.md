# 🎯 Zero-Shot & Foundation Models
## System 06: Foundation Models for 3D Understanding

Production-ready library for integrating foundation models (CLIP, SAM) with 3D point clouds for zero-shot classification, segmentation, and scene understanding.

Part of the **Spatial AI Architect Program** by [3D Geodata Academy](https://learngeodata.eu).

## 🎯 Key Features

- **CLIP Integration**: Zero-shot classification using text-image alignment
- **SAM Integration**: Segment Anything for 3D segmentation
- **Multi-View Rendering**: Convert 3D to 2D for foundation model processing
- **Prompt Engineering**: Optimized prompts for different domains
- **Adapter Networks**: Fine-tune foundation models for 3D
- **Zero-Shot Segmentation**: Semantic and instance segmentation
- **Open-Vocabulary Search**: Query point clouds with natural language

## 📦 Installation

```bash
# Install dependencies
pip install numpy scipy torch torchvision Pillow scikit-learn

# Install CLIP
pip install git+https://github.com/openai/CLIP.git

# Install SAM (optional)
pip install git+https://github.com/facebookresearch/segment-anything.git

# Clone and use
git clone https://github.com/florentPoux/point-cloud-processing.git
cd point-cloud-processing
```

## 🚀 Quick Start

### Zero-Shot Classification

```bash
python zero_shot_foundation/examples/zero_shot_classification.py input.las \
    --classes tree building car ground \
    --domain outdoor
```

### Python API

```python
from zero_shot_foundation.models.clip_3d import (
    load_clip_model,
    zero_shot_classify_point_cloud
)

# Load CLIP
clip_model, preprocess = load_clip_model('ViT-B/32', device='cuda')

# Classify
class_names = ['tree', 'building', 'car', 'ground']
predicted_class, confidence, similarities = zero_shot_classify_point_cloud(
    points, colors, class_names, clip_model, preprocess
)

print(f"Predicted: {class_names[predicted_class]} ({confidence:.2f})")
```

## 📚 Modules

### Models (`models/`)
- `clip_3d.py` - CLIP integration for 3D data
- `sam_3d.py` - Segment Anything for point clouds

### Adapters (`adapters/`)
- `point_adapter.py` - Adapter networks, LoRA, prompt learning

### Prompting (`prompting/`)
- `templates.py` - Domain-specific prompt templates

### Inference (`inference/`)
- `zero_shot_predictor.py` - Zero-shot classification and segmentation

## 🎓 Examples

### CLIP for Classification

```python
from zero_shot_foundation.models.clip_3d import (
    load_clip_model,
    encode_text_with_ensembling,
    encode_point_cloud_with_clip
)

# Load model
clip_model, preprocess = load_clip_model('ViT-B/32')

# Encode text
class_names = ['chair', 'table', 'lamp']
text_features = encode_text_with_ensembling(class_names, clip_model)

# Encode point cloud
point_features = encode_point_cloud_with_clip(
    points, colors, clip_model, preprocess, num_views=8
)

# Classify
similarities = point_features @ text_features.T
predicted = np.argmax(similarities)
```

### Multi-View Rendering

```python
from zero_shot_foundation.models.clip_3d import render_point_cloud_views

# Render from multiple viewpoints
views = render_point_cloud_views(
    points, colors, num_views=8, image_size=224
)

# Each view is a 224x224 RGB image
for i, view in enumerate(views):
    print(f"View {i} shape: {view.shape}")
```

### SAM for Segmentation

```python
from zero_shot_foundation.models.sam_3d import (
    load_sam_model,
    auto_segment_point_cloud,
    interactive_segment_with_clicks
)

# Load SAM
sam, sam_predictor = load_sam_model(
    model_type='vit_h',
    checkpoint_path='sam_vit_h.pth'
)

# Automatic segmentation
segments = auto_segment_point_cloud(
    points, colors, sam_predictor, grid_size=16
)

print(f"Found {len(segments)} segments")

# Interactive segmentation with clicks
positive_clicks = points[[100, 200, 300]]  # User-selected points
labels, score, masks = interactive_segment_with_clicks(
    points, colors, sam_predictor, positive_clicks
)
```

### Prompt Engineering

```python
from zero_shot_foundation.prompting.templates import (
    create_classification_prompts,
    create_part_segmentation_prompts,
    create_hierarchical_prompts
)

# Classification prompts
prompts = create_classification_prompts(
    ['building', 'tree', 'car'],
    domain='outdoor'
)

# Part segmentation prompts
part_prompts = create_part_segmentation_prompts(
    ['seat', 'backrest', 'armrest', 'leg'],
    object_type='chair'
)

# Hierarchical prompts
hierarchy = {
    'furniture': ['chair', 'table', 'desk'],
    'vehicle': ['car', 'truck', 'bus']
}
hier_prompts = create_hierarchical_prompts(hierarchy)
```

### Adapter Training

```python
from zero_shot_foundation.adapters.point_adapter import (
    create_point_cloud_adapter,
    train_adapter_with_clip
)

# Create adapter
adapter = create_point_cloud_adapter(
    input_dim=512, hidden_dim=256, output_dim=512
)

# Train adapter
trained_adapter = train_adapter_with_clip(
    point_features, text_embeddings, labels,
    adapter, num_epochs=50, lr=1e-3
)

# Apply adapter
adapted_features = apply_adapter(new_point_features, trained_adapter)
```

### Zero-Shot Segmentation

```python
from zero_shot_foundation.inference.zero_shot_predictor import (
    zero_shot_point_segmentation,
    zero_shot_instance_segmentation
)

# Semantic segmentation
class_prompts = ['ground', 'vegetation', 'building', 'vehicle']
labels = zero_shot_point_segmentation(
    points, colors, class_prompts, clip_model, preprocess
)

# Instance segmentation (with SAM)
object_prompts = ['tree', 'building', 'car']
instances = zero_shot_instance_segmentation(
    points, colors, object_prompts, clip_model, preprocess, sam_predictor
)
```

### Open-Vocabulary Search

```python
from zero_shot_foundation.inference.zero_shot_predictor import (
    open_vocabulary_search,
    visual_question_answering_3d
)

# Search database with text query
results = open_vocabulary_search(
    "modern office chair with armrests",
    points_database, colors_database,
    clip_model, preprocess, top_k=10
)

# Visual question answering
question = "What is the main object in this scene?"
answer_candidates = ["chair", "table", "lamp", "desk"]
answer, confidence = visual_question_answering_3d(
    points, colors, question, clip_model, preprocess, answer_candidates
)
```

### Scene Graph Generation

```python
from zero_shot_foundation.inference.zero_shot_predictor import semantic_scene_graph

# Generate scene graph
entity_prompts = ['chair', 'table', 'lamp', 'desk']
relation_prompts = ['on', 'under', 'next to', 'above']

scene_graph = semantic_scene_graph(
    points, colors, entity_prompts, relation_prompts,
    clip_model, preprocess
)

# Explore scene graph
for entity in scene_graph['entities']:
    print(f"Entity {entity['id']}: {entity['type']}")

for relation in scene_graph['relations']:
    print(f"{relation['subject']} -> {relation['relation']} -> {relation['object']}")
```

### LoRA Adaptation

```python
from zero_shot_foundation.adapters.point_adapter import (
    create_lora_adapter,
    apply_lora_to_features
)

# Create LoRA adapter
lora = create_lora_adapter(base_dim=512, rank=8, alpha=16)

# Apply LoRA
adapted_features = apply_lora_to_features(
    point_features, base_embedding, lora
)
```

## 🏆 Advanced Features

### Few-Shot Learning

```python
from zero_shot_foundation.adapters.point_adapter import few_shot_adapter_learning

# Few-shot classification
predictions, similarities = few_shot_adapter_learning(
    support_points, support_labels,
    query_points,
    base_encoder,
    num_classes=10
)
```

### Prompt Optimization

```python
from zero_shot_foundation.adapters.point_adapter import (
    create_prompt_learner,
    optimize_prompts
)

# Create learnable prompts
prompt_learner = create_prompt_learner(num_prompts=5, prompt_dim=512)

# Optimize prompts
optimized_prompts = optimize_prompts(
    point_features, labels, prompt_learner,
    num_classes=10, num_epochs=100
)
```

### Progressive Refinement

```python
from zero_shot_foundation.inference.zero_shot_predictor import (
    progressive_refinement_prediction
)

# Multi-scale progressive prediction
predictions, confidences = progressive_refinement_prediction(
    points, colors, class_prompts,
    clip_model, preprocess, num_iterations=3
)
```

## 📖 Resources

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [SAM Paper](https://arxiv.org/abs/2304.02643)
- [3D Geodata Academy](https://learngeodata.eu)
- [Foundation Models Tutorial](https://medium.com/@florentpoux)

## 🛠️ Built With

* [PyTorch](https://pytorch.org/) - Deep learning framework
* [CLIP](https://github.com/openai/CLIP) - Vision-language model
* [SAM](https://github.com/facebookresearch/segment-anything) - Segmentation model
* [NumPy](https://numpy.org/) - Array operations

## 👨‍💻 Author

**Florent Poux** - [3D Geodata Academy](https://learngeodata.eu)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
