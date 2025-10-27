"""
Segment and analyze particles in an NDPI whole-slide image.

Step 3: 
    a. Read the images with segmented particles from data/<ndpi_stem>_level_<level>_below_thresh_<thresh>
    b. Embed each image into a 200x200 pixel canvas, centering the original image
    c. Save the embedded images into data/<ndpi_stem>_level_<level>_below_thresh_<thresh>_embed

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

import cv2
import numpy as np
import os
from pathlib import Path

def embed_image_centered(image, target_size=300):
    """
    Embed an image into a target_size x target_size canvas, centered.
    
    Args:
        image: Input image (numpy array)
        target_size: Size of the output canvas (default: 300)
    
    Returns:
        Embedded image as numpy array
    """
    # Get original image dimensions
    height, width = image.shape[:2]
    
    # Create white canvas
    if len(image.shape) == 3:  # Color image
        canvas = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    else:  # Grayscale image
        canvas = np.full((target_size, target_size), 255, dtype=np.uint8)
    
    # Calculate position to center the image
    start_y = (target_size - height) // 2
    start_x = (target_size - width) // 2
    
    # Ensure the image fits within the canvas
    end_y = start_y + height
    end_x = start_x + width
    
    # Handle edge case where image might be larger than canvas
    if height > target_size or width > target_size:
        # Resize image to fit within canvas while maintaining aspect ratio
        scale = min(target_size / width, target_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Recalculate position
        start_y = (target_size - new_height) // 2
        start_x = (target_size - new_width) // 2
        end_y = start_y + new_height
        end_x = start_x + new_width
    
    # Place the image on the canvas
    canvas[start_y:end_y, start_x:end_x] = image
    
    return canvas

def process_images_in_folder(input_folder, output_folder, target_size=300):
    """
    Process all images in input folder and save embedded versions to output folder.
    
    Args:
        input_folder: Path to folder containing input images
        output_folder: Path to folder where processed images will be saved
        target_size: Size of the output canvas (default: 300)
    """
    # Create input and output Path objects
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # Counter for processed images
    processed_count = 0
    error_count = 0
    
    print(f"Processing images from: {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Target canvas size: {target_size}x{target_size}")
    print("-" * 50)
    
    # Process each file in the input directory
    for file_path in input_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                #print(f"Processing: {file_path.name}")
                
                # Read the image
                image = cv2.imread(str(file_path))
                
                if image is None:
                    print(f"  ❌ Could not read image: {file_path.name}")
                    error_count += 1
                    continue
                
                # Get original dimensions
                height, width = image.shape[:2]
                #print(f"  Original size: {width}x{height}")
                
                # Embed the image
                embedded_image = embed_image_centered(image, target_size)
                
                # Create output filename
                #output_filename = f"embedded_{file_path.name}"
                output_filename = file_path.name
                output_file_path = output_path / output_filename
                
                # Save the embedded image
                success = cv2.imwrite(str(output_file_path), embedded_image)
                
                if success:
                    #print(f"  ✅ Saved: {output_filename}")
                    processed_count += 1
                else:
                    #print(f"  ❌ Failed to save: {output_filename}")
                    error_count += 1
                    
            except Exception as e:
                print(f"  ❌ Error processing {file_path.name}: {str(e)}")
                error_count += 1
    
    print("-" * 50)
    print(f"Processing complete!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Errors encountered: {error_count} images")

def main():

    input_folder = "../data/test_level_1_below_thresh_200"     
    output_folder = "../data/test_level_1_below_thresh_200_embed" 
    target_size = 200                
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"❌ Input folder '{input_folder}' does not exist!")
        print("Please create the folder and add your images, or modify the input_folder path in the script.")
        return
    
    # Process the images
    process_images_in_folder(input_folder, output_folder, target_size)

if __name__ == "__main__":
    main()
