"""
Segment and analyze particles in an NDPI whole-slide image.

Step 1: 
    a. Segment particles using image processing techniques
    b. Segment the particles using SAM2 
    c. Save the segmented particles as individual images in data/<ndpi_stem>_level_<level>

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

import os, sys
from shutil import rmtree
import time
from pathlib import Path
from typing import List
import openslide
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from tqdm import tqdm

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Setup automatic & manual SAM2
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
checkpoint = "../../SAM2/sam2.1_hiera_large.pt" # desktop
checkpoint = "./sam2.1_hiera_large.pt"          # laptop
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2 = build_sam2(model_cfg, checkpoint, device="cuda", apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Maximal number of points to generate masks in SAM2 (too many points lead to CUDA out of memory error)
max_sam2_points = 10
    
def method_1(image: np.ndarray, min_area: int, max_area: int) -> List[np.ndarray]:
    """
    Segment particles using thresholding of a grayscale image after blurring and enhancing contrast.
    
    Args:
        image: OpenCV image
        min_area: Minimum particle area
        max_area: Maximum particle area

    Returns:
        List of contours around particles
    """    

    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # Apply Otsu's thresholding
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations to clean up the binary image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Remove noise
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Fill small holes
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Distance transform for watershed
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)

    # Find local maxima (sure foreground)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Find sure background area
    sure_bg = cv2.dilate(closing, kernel, iterations=3)

    # Find unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add 1 to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply watershed
    image_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(image_color, markers)

    # Find contours of segmented particles
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area to remove noise
    # filtered_contours = [cnt for cnt in contours 
    #                    if min_area < cv2.contourArea(cnt) < max_area]
    filtered_contours = [cnt for cnt in contours 
                if min_area < cv2.contourArea(cnt)]
                
    return filtered_contours                


def method_2(image: np.ndarray, min_area: int, max_area: int) -> List[np.ndarray]:
    """
    Segment particles by replacing a distribution of background colors of a grayscale image
    with a single value at the left edge of this distribution, followed by thresholding. 
    
    Args:
        image: OpenCV image
        min_area: Minimum particle area
        max_area: Maximum particle area

    Returns:
        List of contours around particles
    """    

    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apparently the background color is not uniform but a distribution
    # Get the distribution density of grayscale bringhtnesses
    pixels = gray.flatten()
    # background_color = cv2.mean(gray)[0]
    # background_color = int(np.median(pixels))
    density, bins = np.histogram(pixels, bins=50, density=True)
    # plt.plot(bins[:-1], density)

    # Get the peaks of the distribution and their widths at the specified relative height
    peaks, _ = find_peaks(density)
    widths = peak_widths(density, peaks, rel_height=0.5)
    # plt.plot(density)
    # plt.plot(peaks, density[peaks], 'x')
    # plt.hlines(*widths[1:], color="C3")   # Unpack the rows of widths into y, xmin, xmax required by hlines

    # Get the width of the highest peak
    imax = np.argmax(density[peaks])
    pw = widths[0][imax]

    # Assign background color to the value left from the highest peak
    background_color = int(bins[peaks[imax]] - pw)

    # fig = plt.figure(num=0)
    # plt.hist(pixels, bins=50, density=True)

    # Create binary mask to separate objects from the background
    # Assign white (255) to all pixels exceeding the background color, and invert the selection to select the objects
    _, binary = cv2.threshold(gray, background_color, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological operations to clean up the binary image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # this removes small holes in the objects
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # this removes small noise outside the objects

    # Find contours, keep only necessary points (CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [cnt for cnt in contours 
                        if min_area < cv2.contourArea(cnt)]

    return filtered_contours     


def discard_contours_near_border(contours: List[np.ndarray], image_shape: tuple, band_width: int) -> List[np.ndarray]:
    """
    Discard contours which are within band_width pixels of the image border.
    
    Args:
        contours: List of contours
        image_shape: Shape of the image (height, width)
        band_width: Width of the border band to disgard contours

    Returns:
        Filtered list of contours
    """    

    height, width = image_shape[:2]
    filtered_contours = []
    for c in contours:
        cc = c.reshape([len(c), 2])
        if np.any(cc[:,0] < band_width) or np.any(cc[:,0] > (width - band_width)) or \
           np.any(cc[:,1] < band_width) or np.any(cc[:,1] > (height - band_width)):
            continue
        filtered_contours.append(c)
    return filtered_contours


def plot_rgb_pixels(contours: List[np.ndarray]):
    """
    Plot the RGB values of pixels within the contours in a 3D scatter plot.
    
    Args:
        contours: List of contours
    """

    # Loop through the contours and get the RGB coordinates of pixels within the contours 
    pixels = []
    for c in contours:

        # Create a mask for the current contour
        mask = np.zeros(opencv_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)  # Fill the contour

        # Get bounding box of contour
        x, y, w, h = cv2.boundingRect(c)

        # Extract the pixels within the contour using the cropped mask (faster than on the whole image)
        cropped_mask = mask[y:y+h, x:x+w]
        cropped_region = opencv_image[y:y+h, x:x+w].copy()
        masked_pixels = cv2.bitwise_and(cropped_region, cropped_region, mask=cropped_mask)

        # Get the RGB values of the pixels within the contour
        pix = cropped_region[cropped_mask == 255]

        # Append to the list of pixels
        pixels.append(pix)


    # Convert list of arrays to a single array
    pixels_all = np.vstack(pixels)
    print(f"Collected {pixels_all.shape[0]} pixels from {len(contours)} contours")

    # Downsample to 100,00 pixels for plotting
    indices = np.random.choice(pixels_all.shape[0], size=10000, replace=False)  
    pixels = pixels_all[indices]

    # Scatter plot of RGB values
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2], s=1)
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')

    plt.show(block=True)


def contour_iou(cnt1, cnt2, shape):
    """
    Compute IoU overlap between two contours.
    
    cnt1, cnt2 : contours from cv2.findContours
    shape      : (height, width) of the mask to create
    """
    # Create empty masks
    mask1 = np.zeros(shape, dtype=np.uint8)
    mask2 = np.zeros(shape, dtype=np.uint8)

    # Draw filled contours
    cv2.drawContours(mask1, [cnt1], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(mask2, [cnt2], -1, 255, thickness=cv2.FILLED)

    # Intersection and union
    intersection = cv2.bitwise_and(mask1, mask2)
    union = cv2.bitwise_or(mask1, mask2)

    inter_area = np.count_nonzero(intersection)
    union_area = np.count_nonzero(union)

    if union_area == 0:
        return 0.0  # avoid division by zero

    return inter_area / union_area

    
# Debugging flags
plotting = False
#plotting = True
verbose = False

# Path to the NDPI file
data_path = Path('../data/large')
ndpi_file = "2025-06-18_10.23.51.ndpi"
ndpi_file = "test.ndpi"
ndpi_file = "large.ndpi"

ndpi_stem = ndpi_file.split('.ndpi')[0]

# Open the NDPI file using OpenSlide
ndpi_path = data_path / ndpi_file
try:
    slide = openslide.OpenSlide(ndpi_path)
except Exception as e:
    print(f"Error opening NDPI file: {e}")
    sys.exit(0)

print(f"Opening file: {ndpi_path}") 

# Print slide information
print(f"Slide dimensions: {slide.dimensions}")
print(f"Available levels: {slide.level_count}")
#print(f"Level dimensions: {[slide.level_dimensions[i] for i in range(slide.level_count)]}")    

# Resolution level (original resolution: 0, higher=faster but lower resolution)
# cv2.error: Unknown C++ exception from OpenCV code for level = 0
level = 2
if level >= slide.level_count:
    level = slide.level_count - 1
    print(f"Adjusted level to maximum available: {level}")

# Get image at specified level
level_dimensions = slide.level_dimensions[level]
print(f"Processing at level {level} with dimensions: {level_dimensions}")

# Create a folder for the particles' images
particles_folder = os.path.join(data_path, f'{ndpi_stem}_level_{level}')
if os.path.isdir(particles_folder):
    ans = input('The detected folder ' + particles_folder
                + ' will be deleted!\n'
                    'Press any key to continue or [n] to exit:')
    if ans == 'n':
        sys.exit(0)
    else:
        rmtree(particles_folder, ignore_errors=True)
        time.sleep(10)

try:
    os.mkdir(particles_folder)
except OSError:
    print('Cannot create the folder ' + particles_folder)
    sys.exit(0)
    
# The particles images will be named 00001 etc
n_digits = 6

# Read the entire slide at the specified level
slide_image = slide.read_region((0, 0), level, level_dimensions)

# Convert RGBA to RGB
slide_image = slide_image.convert('RGB')

# Convert PIL image to OpenCV format
opencv_image = cv2.cvtColor(np.array(slide_image), cv2.COLOR_RGB2BGR)

if opencv_image is None:
    raise ValueError(f"Could not convert to OpenCV image..")
else:
    print(f"Loaded OpenCV image with shape: {opencv_image.shape}")

# Close the slide
slide.close()

print("-" * 50)

# Discard the contours which are within band_width pixels of the image border
band_width = 10
height, width = opencv_image.shape[:2]

# Minimum and maximum particle area in pixels for the highest resolution level (=0) 
min_area = 2000  
max_area = 200000 

# Adjust the areas to the selected level
min_area /= 2**(level - 1)  
max_area /= 2**(level - 1)

# Copy the OpenCV image for delineating contours
annotated = opencv_image.copy()

# Method 1: 
#   convert to grayscale
#   Gaussian blur
#   enhance contrast
#   Otsu's thresholding
#   connected contours with areas larger than min_area 
contours_1 = method_1(opencv_image, min_area, max_area)
nc1 = len(contours_1)
print(f"Method 1: detected {nc1} particles")

# Discard the contours which are within band_width pixels of the image border
contours_1 = discard_contours_near_border(contours_1, opencv_image.shape, band_width)
print(f"Method 1: after discarding {nc1 - len(contours_1)} contours near border, {len(contours_1)} particles remain")

# # Draw green contours around detected particles
# for c in contours_1:
#     # Draw the contour on the image
#     cv2.drawContours(annotated, [c], -1, (0, 255, 0), thickness=1)

# Method 2: 
#   convert to grayscale
#   Assign background color to the value left from the highest peak in the distribution of background colors
#   contours with areas larger than min_area 
contours_2 = method_2(opencv_image, min_area, max_area)
nc2 = len(contours_2)
print(f"Method 2: detected {nc2} particles")

# Discard the contours which are within band_width pixels of the image border
contours_2 = discard_contours_near_border(contours_2, opencv_image.shape, band_width)
print(f"Method 2: after discarding {nc2 - len(contours_2)} contours near border, {len(contours_2)} particles remain")

if len(contours_1) > len(contours_2):
    contours = contours_1
    print(f"Using contours from Method 1")
else:
    contours = contours_2
    print(f"Using contours from Method 2")

# # Draw red contours around detected particles
# for c in contours_2:
#     # Draw the contour on the image
#     cv2.drawContours(annotated, [c], -1, (0, 0, 255), thickness=1)

# Copy the OpenCV image for delineating contours
#tmp = opencv_image.copy()

# imfile = 'particles_1_2.png'
# cv2.imwrite(imfile, annotated)
# print(f"Saved the plotted contours around detected particles to {imfile}")

# Limit the number of contours for debugging
#contours = contours[:10]

# Initialize a list for the contours around detected particles
particles_contours = []

# Analyze the color distribution inside of the contours
# Use tqdm to show progress bar 
ic = 0
for n, c in enumerate(tqdm(contours, desc="Segmenting particles")):
#for n, c in enumerate(contours):
    # Try to use thresholding for segmenting particles
    use_thresholding = True

    # Create a mask for the current contour
    mask = np.zeros(opencv_image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)  # Fill the contour

    # Get bounding box of contour
    x, y, w, h = cv2.boundingRect(c)

    # Extract the pixels within the contour using the cropped mask (faster than on the whole image)
    cropped_mask = mask[y:y+h, x:x+w]
    cropped_region = opencv_image[y:y+h, x:x+w].copy()
    masked_pixels = cv2.bitwise_and(cropped_region, cropped_region, mask=cropped_mask)

    # Debugging
    if n >= 338: 
        a = 1

    if plotting:
        cv2.imshow('masked_pixels', masked_pixels)
        #cv2.waitKey(-1) 

    # Thresholding on grayscale
    gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
    pix = gray[cropped_mask == 255]

    density, bins = np.histogram(pix, bins=50, density=True)
    if plotting:
        plt.figure()
        plt.plot(bins[:-1], density)

    # Get the peaks of the distribution... 
    peaks, _ = find_peaks(density)
    if len(peaks) < 1:
        if verbose:
            print("No peaks found in grayscale distribution for contour {ic} of {ndpi_file} → using thresholding")
            print('Keep the mask by thresholding')
    else:
        # ... and their widths at the specified relative height
        widths = peak_widths(density, peaks, rel_height=0.5)
        if plotting:
            plt.plot(bins[:-1][peaks], density[peaks], 'x')

        # Find all peaks which are at least 10% of the highest peak
        ipeaks = np.where(density[peaks] > 0.1 * np.max(density[peaks]))[0]
        if len(ipeaks) < 1:
            if verbose:
                print(f"No significant peaks found in grayscale distribution for {ndpi_file}, contour {ic}")
                print('Keep the mask by thresholding')
        else:
            if len(ipeaks) == 1:
                if verbose:
                    print(f"A single significant peak found in grayscale distribution for contour {ic} of {ndpi_file} → using thresholding")
            else:
                if verbose:
                    print(f"Significant peaks found in grayscale distribution for contour {ic} of {ndpi_file} → using SAM2")
                use_thresholding = False

    
    if use_thresholding:

        # Plot the bounding box around the contour
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 1)  

        # Indictate the contour number
        cv2.putText(annotated, str(ic), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) 

        # Draw red contours around the detected particle
        cv2.drawContours(annotated, [c], -1, (0, 0, 255), thickness=1)

        # Add the contour to the list of detected particles
        particles_contours.append(c)
    
    else:

        if plotting:
            plt.plot(bins[peaks[ipeaks]], density[peaks[ipeaks]], 'o')

        # # Only consider peaks in the small hue range which correspond to brownish colors
        # hue_threshold = 30
        # ind = np.where(bins[peaks[ipeaks]] < hue_threshold)[0]
        # if len(ind) < 1:
        #     continue
        #     # raise ValueError(f"No peaks found in hue distribution for {ndpi_file}, contour {ic}")
        # ipeaks = ipeaks[ind]

        # The corresponding widths of the highest peaks
        pwidths = widths[0][ipeaks]

        particles_coords = []
        for (ip, pw) in zip(ipeaks, pwidths):
            # Select the pixels within the width of the hue peak
            mask = (gray >= bins[:-1][peaks][ip] - pw/2) & (gray <= bins[:-1][peaks][ip] + pw/2)
            
            # Limit the selection to the contour area
            mask = mask & cropped_mask.astype(bool)
            if plotting:
                cv2.imshow('mask', mask.astype(np.uint8) * 255)
            
            # Get coordinates (row indices, col indices) and convert them to a numpy array  
            coords = np.where(mask)
            coords = np.column_stack(coords)

            # In OpenCV, images are accessed in (y, x) order. The indexing starts from the top-left corner, which is (0, 0).
            coords = coords[:, [1, 0]]  # Switch to (x, y) order

            # Points with coords are the inputs to SAM2 for segmenting the particles_coords
            particles_coords.append(coords)

        # If there are black pixels inside of the contour, add them since they are not identified by hue values
        black_threshold = 30
        black_mask = (cropped_region[:, :, 0] < black_threshold) & (cropped_region[:, :, 1] < black_threshold) & (cropped_region[:, :, 2] < black_threshold)
        black_mask = black_mask & cropped_mask.astype(bool)
        coords = np.where(black_mask)
        coords = np.column_stack(coords)
        coords = coords[:, [1, 0]]  # Switch to (x, y) order
        if coords.shape[0] > 0:
            particles_coords.append(coords)

        # Initialize the masks for all segmented particles
        all_masks = np.zeros((h, w), dtype=bool)

        # debug plotting
        tmp = masked_pixels.copy()

        # Convert BGR to RGB for SAM2
        masked_pixels_rgb = cv2.cvtColor(masked_pixels, cv2.COLOR_BGR2RGB)

        # Set the image for the predictor           
        predictor.set_image(masked_pixels_rgb)

        # Accumulate non-intersecting contours for cropped_region
        contours_local = []

        # Generate masks for each point
        for coords in particles_coords:
            for c in coords:
                
                # Do not run SAM2 predictions if coords are already covered by previous masks
                if all_masks[c[1], c[0]]:
                    continue

                # Reshape the current point to a 2D array (needed for pytorch)
                point = c.reshape(1, -1) if c.shape[0] > 1 else c

                # Create labels (=1 i.e. foreground) with the same length as coords
                label = np.ones((point.shape[0],), dtype=int)

                # Get the masks, corresponding to the input points
                masks, scores, _ = predictor.predict(
                    point_coords=point,
                    point_labels=label,
                    box=None,
                    multimask_output=False,
                )   

                # Analyze the only mask
                mask = masks[0] 

                # Characterize the mask by its area and connectivity 
                # num_labels includes the background (label 0), so we need exactly 2 labels for connectivity:
                # - Label 0: background (False pixels)
                # - Label 1: the single connected component (True pixels)                 
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8) * 255)
                try:
                    area = stats[1, cv2.CC_STAT_AREA] 
                except Exception as e:
                    #print(f"Error processing mask for contour {ic} in {ndpi_file}: {e}")
                    # Skip this mask
                    continue

                # Discard masks that are not connected or are too small 
                mask_connected = num_labels == 2
                if not mask_connected or area < min_area:
                    continue      

                # Discard masks that are too small or too large
                # Area of the single connected component

                #tmp[mask.astype(bool)] = [0, 255, 0]  # Mark the selected pixels in green
                # cv2.imshow('im', tmp)

                # Get a contour around the mask
                contour_mask, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contour_mask) > 1:
                    raise ValueError(f"A single contour expected for mask {ic} in {ndpi_file}")
                
                # The contour in the local (cropped) coordinates
                contour_mask = contour_mask[0]

                # Check for overlap with previously accumulated contours
                overlap = False
                for c0 in contours_local:
                    iou = contour_iou(contour_mask, c0, (h, w))
                    if iou > 0.9:
                        overlap = True
                        break
                
                # Discard overlapping contours 
                if overlap:
                    continue

                # Accumulate the local contours
                contours_local.append(contour_mask)

                # The contour in the global (full-image) coordinates
                contour_global = contour_mask + np.array([[[x, y]]])

                # Draw red contours around the detected particle on the cropped image
                cv2.drawContours(masked_pixels, contour_mask, -1, (0, 0, 255), thickness=1)
                if plotting:
                    cv2.imshow('masked_pixels', masked_pixels)
                
                # Get bounding box of contour
                xc, yc, wc, hc = cv2.boundingRect(contour_mask)

                # Plot the bounding box around the contour on the full image
                cv2.rectangle(annotated, (x+xc, y+yc), (x+xc + wc, y+yc + hc), (0, 0, 255), 1)  

                # Indictate the contour number
                cv2.putText(annotated, str(ic), (x+xc, y+yc-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) 

                # Draw red contours around the detected particle on the full image
                cv2.drawContours(annotated, [contour_global], -1, (0, 0, 255), thickness=1)
        
                # Add the contour to the list of detected particles
                particles_contours.append(contour_global)
                                            
                # Accumulate the masks
                all_masks |= mask.astype(bool)

                ic += 1

        # Decrease contour index to keep the numbering consistent with thresholding    
        ic -= 1

    # Increment contour index
    ic += 1

    if plotting:
        cv2.destroyAllWindows()
        plt.close('all')


imfile = f'{ndpi_stem}_particles_level_{level}.png'
impath = os.path.join(data_path, imfile)
cv2.imwrite(impath, annotated)
print(f"Saved the plotted contours around detected particles to {impath}")

print("-" * 50)

# Extract particles from contours and save them as separate image files
for ic, pc in enumerate(tqdm(particles_contours, desc="Saving particles images")):
#for pc in particles_contours:

    # Get bounding box of contour
    xc, yc, wc, hc = cv2.boundingRect(pc)

    # Draw filled contours on a mask, covering the particle
    mask = np.zeros((hc, wc), dtype=np.uint8)
    cv2.drawContours(mask, [pc - np.array([[[xc, yc]]])], -1, 255, thickness=cv2.FILLED) 

    # Keep only the pixels within the contour, set pixels outside of the mask to white
    particle = opencv_image[yc:yc+hc, xc:xc+wc].copy()
    particle[mask == 0] = [255, 255, 255]

    # Write the particle image
    imname = str(ic).rjust(n_digits, '0') + '.png'  
    impath = os.path.join(particles_folder, imname)
    cv2.imwrite(impath, particle)


