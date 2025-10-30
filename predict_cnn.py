"""
Classify particles as spore or non-spore using fine-tuned ResNet18 model.

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
import shutil
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm


MODEL_PATH = "models/spore_classifier_resnet18.pth"
INPUT_FOLDER = '../data/test_level_1_below_thresh_200_embed_class_filter'  # Folder containing particles to classify
SPORES_CSV = "../data/predicted_spores.csv"
NON_SPORES_CSV = "../data/predicted_non_spores.csv"
SPORES_OUTPUT_FOLDER = "../data/spores_predicted"  # Folder to copy predicted spores

IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5  # Probability threshold for spore classification

# ============================================================================
# Model Setup
# ============================================================================

def load_model(model_path, device):
    """Load trained model"""
    
    # Create model architecture (same as training)
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Model trained for {checkpoint['epoch']+1} epochs")
    print(f"Validation accuracy: {checkpoint['val_acc']:.4f}")
    
    return model

def get_transform():
    """Get inference transform (no augmentation)"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ============================================================================
# Inference Functions
# ============================================================================

def predict_image(model, image_path, transform, device):
    """Predict single image"""
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()
    
    return probability

def classify_folder(model, input_folder, transform, device):
    """Classify all images in folder"""
    
    # Get all image files
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) == 0:
        print(f"No images found in {input_folder}")
        return None
    
    print(f"Found {len(image_files)} images to classify")
    
    # Classify each image
    results = []
    for filename in tqdm(image_files, desc="Classifying"):
        image_path = os.path.join(input_folder, filename)
        
        try:
            probability = predict_image(model, image_path, transform, device)
            prediction = "spore" if probability >= CONFIDENCE_THRESHOLD else "non_spore"
            
            results.append({
                'filename': filename,
                'prediction': prediction,
                'spore_probability': probability,
                'confidence': max(probability, 1 - probability)
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results.append({
                'filename': filename,
                'prediction': 'error',
                'spore_probability': None,
                'confidence': None
            })
    
    return pd.DataFrame(results)

def copy_predicted_spores(results_df, input_folder, output_folder):
    """Copy images predicted as spores to output folder"""
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get spore predictions
    spore_predictions = results_df[results_df['prediction'] == 'spore']
    
    if len(spore_predictions) == 0:
        print(f"No spores predicted, {output_folder} folder is empty")
        return
    
    # Copy files
    print(f"\nCopying {len(spore_predictions)} predicted spores to {output_folder}...")
    for _, row in tqdm(spore_predictions.iterrows(), total=len(spore_predictions), desc="Copying"):
        src = os.path.join(input_folder, row['filename'])
        dst = os.path.join(output_folder, row['filename'])
        
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"Error copying {row['filename']}: {e}")
    
    print(f"Copied {len(spore_predictions)} images to {output_folder}")

# ============================================================================
# Main
# ============================================================================

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    model = load_model(MODEL_PATH, device)
    
    # Get transform
    transform = get_transform()
    
    # Classify images
    print(f"\nClassifying images from {INPUT_FOLDER}...")
    results_df = classify_folder(model, INPUT_FOLDER, transform, device)
    
    if results_df is not None:
        spores_df = results_df[results_df['prediction'] == 'spore'].copy()
        non_spores_df = results_df[results_df['prediction'] == 'non_spore'].copy()
        
        # Save spores CSV
        if len(spores_df) > 0:
            spores_df.to_csv(SPORES_CSV, index=False)
            print(f"\nSpore predictions saved to {SPORES_CSV}")
        else:
            print(f"\nNo spores predicted, {SPORES_CSV} not created")
        
        # Save non-spores CSV
        if len(non_spores_df) > 0:
            non_spores_df.to_csv(NON_SPORES_CSV, index=False)
            print(f"Non-spore predictions saved to {NON_SPORES_CSV}")
        else:
            print(f"No non-spores predicted, {NON_SPORES_CSV} not created")
        
        copy_predicted_spores(results_df, INPUT_FOLDER, SPORES_OUTPUT_FOLDER)
        
        # Print summary
        print("\n" + "="*60)
        print("Classification Summary")
        print("="*60)
        print(f"Total images: {len(results_df)}")
        print(f"Predicted spores: {len(spores_df)}")
        print(f"Predicted non-spores: {len(non_spores_df)}")
        print(f"Errors: {(results_df['prediction'] == 'error').sum()}")
        
        # Show high-confidence predictions
        if len(spores_df) > 0:
            print("\nHigh-confidence spore predictions (>0.9):")
            high_conf_spores = spores_df[
                spores_df['spore_probability'] > 0.9
            ].sort_values('spore_probability', ascending=False)
            
            if len(high_conf_spores) > 0:
                print(high_conf_spores[['filename', 'spore_probability']].head(10).to_string(index=False))
            else:
                print("  None found")
        
        if len(non_spores_df) > 0:
            print("\nHigh-confidence non-spore predictions (>0.9):")
            high_conf_non_spores = non_spores_df[
                non_spores_df['confidence'] > 0.9
            ].sort_values('confidence', ascending=False)
            
            if len(high_conf_non_spores) > 0:
                print(high_conf_non_spores[['filename', 'spore_probability']].head(10).to_string(index=False))
            else:
                print("  None found")

if __name__ == "__main__":
    main()
