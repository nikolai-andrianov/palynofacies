"""
Fine-tune ResNet18 for spore vs non-spore classification.

Copyright 2025 Nikolai Andrianov, nia@geus.dk

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


SPORES_FOLDER = "../data/spores"
NON_SPORES_FOLDER = "../data/non_spores"
OUTPUT_MODEL_PATH = "models/spore_classifier_resnet18.pth"
RESULTS_FOLDER = "training_results"

# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE_FINETUNE = 1e-4  # For layer3, layer4
LEARNING_RATE_FC = 1e-3        # For final classification layer
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10
RANDOM_SEED = 42

# Image settings
IMAGE_SIZE = 224  # ResNet expects 224x224
NUM_WORKERS = 4

# ============================================================================
# Dataset Class
# ============================================================================

class ParticleDataset(Dataset):
    """Dataset for particle images with augmentation"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# Data Loading and Augmentation
# ============================================================================

def load_dataset():
    """Load images from spores and non_spores folders"""
    
    image_paths = []
    labels = []
    
    # Load spores (label = 1)
    spore_files = [f for f in os.listdir(SPORES_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for filename in spore_files:
        image_paths.append(os.path.join(SPORES_FOLDER, filename))
        labels.append(1)
    
    # Load non-spores (label = 0)
    non_spore_files = [f for f in os.listdir(NON_SPORES_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for filename in non_spore_files:
        image_paths.append(os.path.join(NON_SPORES_FOLDER, filename))
        labels.append(0)
    
    print(f"Loaded {len(spore_files)} spores and {len(non_spore_files)} non-spores")
    print(f"Total dataset size: {len(image_paths)} images")
    
    return image_paths, labels

def get_transforms():
    """Define data augmentation transforms"""
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(180),  # Full rotation for circular particles
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    # Validation transforms without augmentation
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# ============================================================================
# Model Setup
# ============================================================================

def create_model():
    """Create ResNet18 with fine-tuning strategy"""
    
    # Load pre-trained ResNet18
    model = models.resnet18(pretrained=True)
    
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.bn1.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # Binary classification (sigmoid output)
    
    return model

# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track metrics
            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels

# ============================================================================
# Visualization Functions
# ============================================================================

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training and validation metrics"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc', marker='o')
    ax2.plot(val_accs, label='Val Acc', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'training_history.png'), dpi=150)
    print(f"Saved training history plot to {RESULTS_FOLDER}/training_history.png")

def plot_confusion_matrix(labels, preds):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Spore', 'Spore'],
                yticklabels=['Non-Spore', 'Spore'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'confusion_matrix.png'), dpi=150)
    print(f"Saved confusion matrix to {RESULTS_FOLDER}/confusion_matrix.png")

# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Create output directories
    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    image_paths, labels = load_dataset()
    
    # Split into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=VALIDATION_SPLIT, 
        random_state=RANDOM_SEED, stratify=labels
    )
    
    print(f"Train set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = ParticleDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = ParticleDataset(val_paths, val_labels, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = create_model()
    model = model.to(device)
    
    finetune_params = list(model.layer3.parameters()) + list(model.layer4.parameters())
    fc_params = list(model.fc.parameters())
    
    optimizer = optim.Adam([
        {'params': finetune_params, 'lr': LEARNING_RATE_FINETUNE},
        {'params': fc_params, 'lr': LEARNING_RATE_FC}
    ])
    
    # Loss function with class weighting for imbalanced data
    criterion = nn.BCEWithLogitsLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Track history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, OUTPUT_MODEL_PATH)
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Final evaluation
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    # Load best model for final evaluation
    checkpoint = torch.load(OUTPUT_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation metrics
    val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')
    
    print(f"\nFinal Validation Metrics:")
    print(f"  Accuracy:  {val_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Plot results
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(val_labels, val_preds)
    
    print(f"\nModel saved to: {OUTPUT_MODEL_PATH}")
    print(f"Results saved to: {RESULTS_FOLDER}/")

if __name__ == "__main__":
    main()
