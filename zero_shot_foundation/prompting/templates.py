import numpy as np

def create_classification_prompts(class_names, domain='general'):
    """Create optimized prompts for classification"""
    if domain == 'general':
        templates = [
            "a photo of a {}",
            "a 3D model of a {}",
            "a point cloud showing a {}",
            "a scan of a {}"
        ]
    elif domain == 'indoor':
        templates = [
            "an indoor scene containing a {}",
            "a {} in a room",
            "furniture piece: {}",
            "indoor object: {}"
        ]
    elif domain == 'outdoor':
        templates = [
            "an outdoor scene with a {}",
            "urban element: {}",
            "a {} in the environment",
            "architectural structure: {}"
        ]
    elif domain == 'building':
        templates = [
            "building component: {}",
            "architectural element: {}",
            "a {} part of a building",
            "construction element: {}"
        ]
    else:
        templates = ["a {}"]

    all_prompts = []
    for class_name in class_names:
        class_prompts = [template.format(class_name) for template in templates]
        all_prompts.append(class_prompts)

    return all_prompts

def create_part_segmentation_prompts(part_names, object_type):
    """Create prompts for part segmentation"""
    templates = [
        f"the {{}} of a {object_type}",
        f"{{}} part of a {object_type}",
        f"a {object_type}'s {{}}",
        f"{object_type} component: {{}}"
    ]

    all_prompts = []
    for part_name in part_names:
        part_prompts = [template.format(part_name) for template in templates]
        all_prompts.append(part_prompts)

    return all_prompts

def create_scene_understanding_prompts(scene_types):
    """Create prompts for scene understanding"""
    templates = [
        "a 3D scene of {}",
        "an environment showing {}",
        "{} scene type",
        "a place that is {}"
    ]

    all_prompts = []
    for scene_type in scene_types:
        scene_prompts = [template.format(scene_type) for template in templates]
        all_prompts.append(scene_prompts)

    return all_prompts

def create_attribute_prompts(attributes, object_name):
    """Create prompts for attribute detection"""
    templates = [
        f"a {{}} {object_name}",
        f"a {object_name} that is {{}}",
        f"{object_name} with {{}} property",
        f"{{}} {object_name} object"
    ]

    all_prompts = []
    for attribute in attributes:
        attr_prompts = [template.format(attribute) for template in templates]
        all_prompts.append(attr_prompts)

    return all_prompts

def create_hierarchical_prompts(class_hierarchy):
    """Create hierarchical prompts from taxonomy"""
    prompts = {}

    for parent, children in class_hierarchy.items():
        prompts[parent] = [
            f"a {parent}",
            f"category: {parent}",
            f"{parent} object"
        ]

        for child in children:
            prompts[child] = [
                f"a {child}, which is a type of {parent}",
                f"{child} ({parent} category)",
                f"a specific {parent}: {child}"
            ]

    return prompts

def create_compositional_prompts(object_parts, spatial_relations=None):
    """Create compositional prompts describing object structure"""
    if spatial_relations is None:
        spatial_relations = ['above', 'below', 'next to', 'inside', 'on top of']

    prompts = []

    for i, part1 in enumerate(object_parts):
        for part2 in object_parts[i+1:]:
            for relation in spatial_relations:
                prompt = f"{part1} {relation} {part2}"
                prompts.append(prompt)

    return prompts

def optimize_prompt_with_examples(base_prompt, positive_examples, negative_examples):
    """Optimize prompt based on examples"""
    modifiers = [
        'detailed', 'clear', 'high-quality', 'realistic',
        'accurate', 'precise', 'distinct', 'well-defined'
    ]

    best_prompt = base_prompt
    best_score = 0

    for modifier in modifiers:
        test_prompt = f"{modifier} {base_prompt}"

        score = evaluate_prompt_quality(test_prompt, positive_examples, negative_examples)

        if score > best_score:
            best_score = score
            best_prompt = test_prompt

    return best_prompt, best_score

def evaluate_prompt_quality(prompt, positive_examples, negative_examples):
    """Evaluate prompt quality based on examples"""
    positive_score = np.mean([similarity(prompt, ex) for ex in positive_examples])
    negative_score = np.mean([similarity(prompt, ex) for ex in negative_examples])

    return positive_score - negative_score

def similarity(text1, text2):
    """Simple text similarity (placeholder for actual embedding similarity)"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union) if len(union) > 0 else 0

def create_context_aware_prompts(class_name, context_info):
    """Create prompts with contextual information"""
    prompts = [f"a {class_name}"]

    if 'location' in context_info:
        prompts.append(f"a {class_name} in {context_info['location']}")

    if 'function' in context_info:
        prompts.append(f"a {class_name} used for {context_info['function']}")

    if 'material' in context_info:
        prompts.append(f"a {class_name} made of {context_info['material']}")

    if 'scale' in context_info:
        prompts.append(f"a {context_info['scale']} {class_name}")

    return prompts

def create_chain_of_thought_prompts(task_description):
    """Create chain-of-thought prompts for complex reasoning"""
    prompts = [
        f"Let's think step by step about {task_description}:",
        f"First, identify the main components.",
        f"Second, analyze their spatial relationships.",
        f"Third, classify based on observed features.",
        f"Finally, provide the classification result."
    ]

    return " ".join(prompts)

def create_few_shot_prompts(class_name, examples):
    """Create few-shot prompts with examples"""
    prompt_parts = [f"Here are examples of {class_name}:"]

    for i, example_desc in enumerate(examples, 1):
        prompt_parts.append(f"Example {i}: {example_desc}")

    prompt_parts.append(f"Based on these examples, identify: {class_name}")

    return " ".join(prompt_parts)
