"""
Segment and analyze particles in an NDPI whole-slide image.

Step 2: 
    a. Read the images with segmented particles from data/<ndpi_stem>_level_<level>
    b. Copy the images into two folders based on a resolution threshold (e.g., 200 pixels)
       - data/<ndpi_stem>_level_<level>_above_thresh_<thresh>
       - data/<ndpi_stem>_level_<level>_below_thresh_<thresh>

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
from PIL import Image
import matplotlib.pyplot as plt

data_dir = '../data'
image_dir = 'test_level_1'
#image_dir = 'test_level_1_below_thresh_300'
image_path = os.path.join(data_dir, image_dir)
print(f'Scanning {image_path}')

# Resolution threshold based on the histograms
thresh = 200

# Number of files larger than thresh
n_large = 0

# Create folder for the images with resolution above and below threshold
large_images_folder = os.path.join(data_dir, f'{image_dir}_above_thresh_{thresh}')
small_images_folder = os.path.join(data_dir, f'{image_dir}_below_thresh_{thresh}')
for folder in [large_images_folder, small_images_folder]:
    if os.path.isdir(folder):
        ans = input('The detected folder ' + folder
                    + ' will be deleted!\n'
                        'Press any key to continue or [n] to exit:')
        if ans == 'n':
            sys.exit(0)
        else:
            rmtree(folder, ignore_errors=True)
            time.sleep(10)

    try:
        os.mkdir(folder)
    except OSError:
        print('Cannot create the folder ' + folder)
        sys.exit(0)
        

max_width, max_height = 0, 0
min_width, min_height = 100000, 100000
fname_maxwidth = ''
fname_maxheight = ''
fname_minwidth = ''
fname_minheight = ''
widths = []
heights = []
n_images = 0

for root, dirs, files in os.walk(image_path):
    for file in files:
        if 'png' in file:
            n_images += 1
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    width, height = img.size

                    widths.append(width)
                    heights.append(height)

                    if width > max_width:
                        max_width = width
                        fname_maxwidth = file
                    if height > max_height:
                        max_height = height
                        fname_maxheight = file

                    if width < min_width:
                        min_width = width
                        fname_minwidth = file
                    if height < min_height:
                        min_height = height
                        fname_minheight = file                        
                    
                    if width > thresh or height > thresh:
                        dest_path = os.path.join(large_images_folder, file)
                        img.save(dest_path)
                        n_large += 1
                    else:
                        dest_path = os.path.join(small_images_folder, file)
                        img.save(dest_path)

                    #print(f"{file_path}: {width}x{height}")
            except Exception as e:
                print(f"Could not open {file_path}: {e}")

print(f'Max width = {max_width}, max height = {max_height}')   
print(f'fname_maxwidth = {fname_maxwidth}') 
print(f'max_height = {fname_maxheight}')     

print(f'Min width = {min_width}, min height = {min_height}')   
print(f'fname_minwidth = {fname_minwidth}') 
print(f'min_height = {fname_minheight}')  

print(f'{n_large} large images copied to {large_images_folder} ({n_large/n_images*100:.2f}%) of the total of {n_images} images')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(widths, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Widths')
plt.xlabel('Width (pixels)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(heights, bins=30, color='salmon', edgecolor='black')
plt.title('Histogram of Heights')
plt.xlabel('Height (pixels)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()