import numpy as np
import torch
import torch.nn as nn

def create_point_cloud_adapter(input_dim=512, hidden_dim=256, output_dim=512):
    """Create adapter network to map point features to foundation model space"""
    adapter = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, output_dim)
    )

    return adapter

def train_adapter_with_clip(point_features, text_embeddings, labels,
                            adapter, num_epochs=50, lr=1e-3, device='cuda'):
    """Train adapter to align point features with CLIP text embeddings"""
    adapter = adapter.to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    point_features = torch.from_numpy(point_features).float().to(device)
    text_embeddings = torch.from_numpy(text_embeddings).float().to(device)
    labels = torch.from_numpy(labels).long().to(device)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        adapted_features = adapter(point_features)
        adapted_features = adapted_features / adapted_features.norm(dim=-1, keepdim=True)

        logits = adapted_features @ text_embeddings.T
        logits = logits * 100

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == labels).float().mean().item()
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")

    return adapter

def apply_adapter(point_features, adapter, device='cuda'):
    """Apply trained adapter to point features"""
    adapter.eval()

    point_features_tensor = torch.from_numpy(point_features).float().to(device)

    with torch.no_grad():
        adapted_features = adapter(point_features_tensor)
        adapted_features = adapted_features / adapted_features.norm(dim=-1, keepdim=True)

    return adapted_features.cpu().numpy()

def create_lora_adapter(base_dim=512, rank=8, alpha=16):
    """Create LoRA adapter for efficient fine-tuning"""
    class LoRAAdapter(nn.Module):
        def __init__(self, dim, rank, alpha):
            super().__init__()
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank

            self.lora_A = nn.Linear(dim, rank, bias=False)
            self.lora_B = nn.Linear(rank, dim, bias=False)

            nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

        def forward(self, x):
            return self.lora_B(self.lora_A(x)) * self.scaling

    return LoRAAdapter(base_dim, rank, alpha)

def apply_lora_to_features(features, base_embedding, lora_adapter, device='cuda'):
    """Apply LoRA adapter on top of base embedding"""
    lora_adapter.eval()

    features_tensor = torch.from_numpy(features).float().to(device)
    base_tensor = torch.from_numpy(base_embedding).float().to(device)

    with torch.no_grad():
        delta = lora_adapter(features_tensor)
        adapted = base_tensor + delta
        adapted = adapted / adapted.norm(dim=-1, keepdim=True)

    return adapted.cpu().numpy()

def few_shot_adapter_learning(support_points, support_labels, query_points,
                               base_encoder, num_classes, device='cuda'):
    """Few-shot learning with adapter on foundation model"""
    adapter = create_point_cloud_adapter(input_dim=512, output_dim=512)
    adapter = adapter.to(device)

    support_features = base_encoder(support_points)
    query_features = base_encoder(query_points)

    class_prototypes = []
    for class_id in range(num_classes):
        class_mask = support_labels == class_id
        class_features = support_features[class_mask]

        adapted_features = adapter(torch.from_numpy(class_features).float().to(device))
        prototype = adapted_features.mean(dim=0)
        prototype = prototype / prototype.norm()

        class_prototypes.append(prototype.cpu().numpy())

    class_prototypes = np.vstack(class_prototypes)

    query_adapted = adapter(torch.from_numpy(query_features).float().to(device))
    query_adapted = query_adapted / query_adapted.norm(dim=-1, keepdim=True)

    similarities = query_adapted.cpu().numpy() @ class_prototypes.T
    predictions = np.argmax(similarities, axis=1)

    return predictions, similarities

def create_prompt_learner(num_prompts=4, prompt_dim=512, init_prompts=None):
    """Create learnable prompt embeddings"""
    class PromptLearner(nn.Module):
        def __init__(self, num_prompts, prompt_dim, init_prompts):
            super().__init__()

            if init_prompts is not None:
                self.prompts = nn.Parameter(init_prompts.clone())
            else:
                self.prompts = nn.Parameter(torch.randn(num_prompts, prompt_dim))

        def forward(self):
            return self.prompts / self.prompts.norm(dim=-1, keepdim=True)

    if init_prompts is not None:
        init_prompts = torch.from_numpy(init_prompts).float()

    return PromptLearner(num_prompts, prompt_dim, init_prompts)

def optimize_prompts(point_features, labels, prompt_learner, num_classes,
                     num_epochs=100, lr=0.001, device='cuda'):
    """Optimize learnable prompts for classification"""
    prompt_learner = prompt_learner.to(device)
    optimizer = torch.optim.Adam(prompt_learner.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    point_features = torch.from_numpy(point_features).float().to(device)
    labels = torch.from_numpy(labels).long().to(device)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        prompts = prompt_learner()

        logits = point_features @ prompts.T
        logits = logits * 100

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == labels).float().mean().item()
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")

    return prompt_learner

def domain_adaptation_with_adapter(source_features, source_labels,
                                   target_features, adapter, num_epochs=50,
                                   lr=1e-4, device='cuda'):
    """Domain adaptation using adapter network"""
    adapter = adapter.to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)

    source_features = torch.from_numpy(source_features).float().to(device)
    source_labels = torch.from_numpy(source_labels).long().to(device)
    target_features = torch.from_numpy(target_features).float().to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.BCEWithLogitsLoss()

    domain_classifier = nn.Linear(512, 1).to(device)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        adapted_source = adapter(source_features)
        adapted_target = adapter(target_features)

        source_domain_pred = domain_classifier(adapted_source)
        target_domain_pred = domain_classifier(adapted_target)

        source_domain_labels = torch.zeros(len(source_features), 1, device=device)
        target_domain_labels = torch.ones(len(target_features), 1, device=device)

        domain_loss = (
            criterion_domain(source_domain_pred, source_domain_labels) +
            criterion_domain(target_domain_pred, target_domain_labels)
        )

        loss = domain_loss

        loss.backward()
        optimizer.step()

    return adapter
