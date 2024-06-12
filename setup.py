from setuptools import setup, find_packages

setup(
    name='WSI_preprocessing',
    version='0.1',  
    packages=find_packages(), 
    url='https://github.com/lolmomarchal/WSI_preprocessing',  
    license='',  
    author='lolmomarchal',
    author_email='lolmomarchal@ucsd.edu',
    description='Used for preprocessing whole slide images',
    install_requires=[
        'h5py==3.11.0',
        'matplotlib==3.8.3',
        'numpy==1.26.4',
        'opencv-python==4.8.0.76',
        'opencv-python-headless==4.9.0.80',
        'openslide-python==1.3.1',
        'pandas==2.2.2',
        'Pillow==10.3.0',
        'scikit-image==0.18.3',
        'torch==2.1.2',
        'torchvision==0.16.2'
    ]  
)
