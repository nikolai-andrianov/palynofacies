# Segment and classify particles in NDPI images

Codes for the project on Automatic Classification of Sedimentary Particles for Insights into Past Climate Environment

# Set up virtual environment

Create a virtual environment with the latest python version
```
conda create --name palyno python
```
Activate the environment in the Anaconda prompt (available in Windows Terminal)
```
conda activate palyno
```

# Install packages
In a clean folder (e.g., in `C:\Trainings\ImageAnalysis\\Palynoscan`):

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install openslide-python openslide-bin
pip install opencv-python
pip install matplotlib
pip install scipy
pip install ultralytics
pip install pandas
pip install tqdm
pip install seaborn
cd C:\Trainings\ImageAnalysis\sam2
pip install -e .
```

# Workflow
0. Edit the files as needed to run a desired NDPI image.
1. Segment particles using image processing techniques and with SAM2 using `analyze_ndpi.py`.
2. Select particles' images based based on a resolution threshold using `particles_image_res.py`.
3. Embed each image into a 200x200 pixel canvas using `embed_images.py`.
4. Classify each image into three categories (black, gray, filter) using `particle_classifier.py`.
5. Extract DINOv2 features from each image and perform k-means clustering using `dino_kmeans.py`.
6. Visualize first images from each cluster using `visualize_clusters.py`.
