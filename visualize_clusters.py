"""
Segment and analyze particles in an NDPI whole-slide image.

Step 6: 
    a. Visualize the first 5 images from each cluster folder in data/<ndpi_stem>_level_<level>_below_thresh_<thresh>_embed_class_filter_clusters_dino

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
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def visualize_cluster_images(base_path=".", num_clusters=8, images_per_cluster=5):
    """
    Visualize the first 5 images from each cluster folder.
    
    Args:
        base_path: Path to the directory containing cluster folders
        num_clusters: Number of cluster folders (0 to num_clusters-1)
        images_per_cluster: Number of images to display per cluster
    """
    fig, axes = plt.subplots(images_per_cluster, num_clusters, 
                             figsize=(24, 15))
    
    # Iterate through each cluster
    for cluster_idx in range(num_clusters):
        cluster_folder = os.path.join(base_path, f"cluster_{cluster_idx}")
        
        # Check if cluster folder exists
        if not os.path.exists(cluster_folder):
            print(f"Warning: {cluster_folder} does not exist")
            continue
        
        # Get all image files from the cluster folder
        image_files = sorted([
            f for f in os.listdir(cluster_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ])
        
        # Display first 5 images
        for img_idx in range(images_per_cluster):
            ax = axes[img_idx, cluster_idx]
            
            if img_idx < len(image_files):
                # Load and display image
                img_path = os.path.join(cluster_folder, image_files[img_idx])
                try:
                    img = Image.open(img_path)
                    ax.imshow(img)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    ax.text(0.5, 0.5, 'Error', ha='center', va='center')
            else:
                # No image available
                ax.text(0.5, 0.5, 'No image', ha='center', va='center')
            
            # Remove axis ticks
            ax.axis('off')
            
            if img_idx == 0:
                ax.set_title(f'Cluster {cluster_idx}', 
                           fontsize=20, 
                           fontweight='bold',
                           pad=10)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # Save the plot
    output_path = os.path.join(base_path, "cluster_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    # Run the visualization
    # Adjust base_path if your cluster folders are in a different location
    visualize_cluster_images(base_path="../data/test_level_1_below_thresh_200_embed_class_filter_clusters_dino")
