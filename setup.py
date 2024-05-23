from setuptools import setup, find_packages

setup(
    name='ddetect',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here
        "coverage==7.4.1",
        "ipykernel==6.22.0",
        "joblib==1.2.0",
        "matplotlib==3.6.3",
        "multidict==6.0.4",
        "numpy==1.22.4",
        "pillow==9.2.0",
        "ptyprocess",
        "pyod==1.0.7",
        "pytest==7.1.2",
        "scikit-learn==1.2",
        "scipy>=1.7.2",
        "torch==1.13.1",
        "torchvision==0.14.1",
    ],
    dependency_links = [
        "git+git://github.com/thelahunginjeet/pyrankagg.git",
        "git+git://github.com/thelahunginjeet/kbutil.git"
    ],
    entry_points={
        'console_scripts': [
            # Add any console scripts or command-line entry points here
        ],
    },
    author='Louis McConnell',
    author_email='mcconnl3@gene.com',
    description='Data Detective is an open-source, modular, extensible validation framework for identifying potential issues with heterogeneous, multimodal data.',
    long_description='In an ideal machine learning setting where data is homogeneous and unimodal, \
users face a variety of challenges in the data preprocessing stages that can quietly \
compromise the performance of a trained model. Issues around data redundancy, \
distribution shift, and anomalous data points will not prevent a model from training, \
but will ultimately hinder performance at test time. Discovering these “silent \
issues” requires a tremendous amount of manual effort and insight, and even then, \
the broad scope of data-related issues can make discovery of these issues infeasible. \
In practical settings, the situation is even more challenging; data often comes \
from a variety of sources, contains noise and outliers, and exhibits significant \
shifts due to exogenous noise. In these cases, manual discovery of data-related \
issues becomes completely infeasible. In light of these settings, we propose Data \
Detective, an open-source, modular, extensible validation framework for identifying \
potential issues with heterogeneous, multimodal data. We empirically investigate \
and test this framework in a case study and show its effectiveness in the \
identification of potential data-related issues.',
    long_description_content_type='text/markdown',
    url='https://github.com/gred-ecdi/datadetective',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
)