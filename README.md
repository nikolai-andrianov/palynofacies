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
