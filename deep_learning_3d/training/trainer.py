import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

def create_optimizer(parameters, optimizer_type='adam', lr=0.001, weight_decay=1e-4):
    """Create optimizer for training"""
    if optimizer_type == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

def create_scheduler(optimizer, scheduler_type='step', **kwargs):
    """Create learning rate scheduler"""
    if scheduler_type == 'step':
        step_size = kwargs.get('step_size', 20)
        gamma = kwargs.get('gamma', 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_type == 'cosine':
        T_max = kwargs.get('T_max', 100)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    elif scheduler_type == 'plateau':
        patience = kwargs.get('patience', 10)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.5)

    else:
        return None

def train_epoch_classification(model_forward_fn, model_components, train_loader, optimizer, device):
    """Train one epoch for classification"""
    total_loss = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()

        logits, transforms = model_forward_fn(data, model_components)

        loss = criterion(logits, labels)

        if transforms:
            from deep_learning_3d.models.pointnet import feature_transform_regularizer
            loss += feature_transform_regularizer(transforms, reg_weight=0.001)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy

def validate_classification(model_forward_fn, model_components, val_loader, device):
    """Validate classification model"""
    total_loss = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)

            logits, _ = model_forward_fn(data, model_components)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy

def train_epoch_segmentation(model_forward_fn, model_components, train_loader, optimizer, device):
    """Train one epoch for segmentation"""
    total_loss = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()

        output, transforms = model_forward_fn(data, model_components)

        output = output.view(-1, output.size(-1))
        labels = labels.view(-1)

        loss = criterion(output, labels.long())

        if transforms:
            from deep_learning_3d.models.pointnet import feature_transform_regularizer
            loss += feature_transform_regularizer(transforms, reg_weight=0.001)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy

def validate_segmentation(model_forward_fn, model_components, val_loader, device):
    """Validate segmentation model"""
    total_loss = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)

            output, _ = model_forward_fn(data, model_components)

            output = output.view(-1, output.size(-1))
            labels = labels.view(-1)

            loss = criterion(output, labels.long())
            total_loss += loss.item()

            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy

def save_checkpoint(filepath, model_components, optimizer, epoch, best_acc):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'best_acc': best_acc,
        'optimizer_state': optimizer.state_dict()
    }

    for name, component in model_components.items():
        if isinstance(component, nn.Module):
            checkpoint[f'{name}_state'] = component.state_dict()
        elif isinstance(component, dict):
            checkpoint[name] = {}
            for sub_name, sub_component in component.items():
                if isinstance(sub_component, nn.Module):
                    checkpoint[name][f'{sub_name}_state'] = sub_component.state_dict()

    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model_components, optimizer=None):
    """Load training checkpoint"""
    checkpoint = torch.load(filepath)

    for name, component in model_components.items():
        if isinstance(component, nn.Module) and f'{name}_state' in checkpoint:
            component.load_state_dict(checkpoint[f'{name}_state'])
        elif isinstance(component, dict) and name in checkpoint:
            for sub_name, sub_component in component.items():
                if isinstance(sub_component, nn.Module) and f'{sub_name}_state' in checkpoint[name]:
                    sub_component.load_state_dict(checkpoint[name][f'{sub_name}_state'])

    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    return checkpoint.get('epoch', 0), checkpoint.get('best_acc', 0.0)

def train_model(model_forward_fn, model_components, train_loader, val_loader,
                num_epochs=100, device='cuda', task='classification',
                save_dir='./checkpoints', lr=0.001):
    """Complete training loop"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    all_parameters = []
    for component in model_components.values():
        if isinstance(component, nn.Module):
            all_parameters.extend(list(component.parameters()))
        elif isinstance(component, dict):
            for sub_component in component.values():
                if isinstance(sub_component, nn.Module):
                    all_parameters.extend(list(sub_component.parameters()))

    optimizer = create_optimizer(all_parameters, lr=lr)
    scheduler = create_scheduler(optimizer, scheduler_type='step', step_size=20, gamma=0.5)

    best_acc = 0.0

    for epoch in range(num_epochs):
        if task == 'classification':
            train_loss, train_acc = train_epoch_classification(
                model_forward_fn, model_components, train_loader, optimizer, device
            )
            val_loss, val_acc = validate_classification(
                model_forward_fn, model_components, val_loader, device
            )
        else:
            train_loss, train_acc = train_epoch_segmentation(
                model_forward_fn, model_components, train_loader, optimizer, device
            )
            val_loss, val_acc = validate_segmentation(
                model_forward_fn, model_components, val_loader, device
            )

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                Path(save_dir) / 'best_model.pth',
                model_components, optimizer, epoch, best_acc
            )

        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                Path(save_dir) / f'checkpoint_epoch_{epoch+1}.pth',
                model_components, optimizer, epoch, best_acc
            )

    print(f"\nTraining complete! Best validation accuracy: {best_acc:.2f}%")

    return best_acc
