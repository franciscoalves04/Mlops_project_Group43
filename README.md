# Project Description: MLOps Pipeline for Retinal Disease Classification 

In this project, we assume the role of MLOps engineers at a start-up company tasked with developing a machine learning solution for eye disease classification from retinal images. The company’s primary objective is to rapidly establish a reliable and production-ready MLOps pipeline rather than to optimize model performance.

The task is to design and implement an end-to-end Machine Learning Operations (MLOps) pipeline that enables fast iteration, scalability, and reproducibility. The pipeline covers the full lifecycle of a machine learning system, including data ingestion, preprocessing, model training, evaluation, versioning and deployment. Emphasis is placed on automation, reproducibility and maintainability, reflecting real-world constraints commonly faced by start-ups.

Eye disease classification serves as a concrete business and medical imaging use case to demonstrate and validate the MLOps pipeline. The resulting system will serve as a solid foundation for experimentation, collaboration, and future production deployment, aligned with industry best practices in MLOps and computer vision.

---

## Dataset
* **Title**: Eye Diseases Classification  
* **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data)  
* **Description**: The dataset contains retinal fundus images across four diagnostic categories: Normal, Diabetic Retinopathy, Cataract and Glaucoma.  
* **Number of samples**: 4,217 images  
* **Size**: 745 MB  

The dataset will be balanced to ensure equal representation of each class, improving model performance and reliability. Images are sourced from multiple high-quality repositories, including IDRiD, HRF and Ocular Recognition databases.

---

## Framework
The pipeline leverages PyTorch and Torchvision for deep learning model development, utilizing pre-trained models, data augmentation utilities and best practices for computer vision.

---

## Model Architecture
The primary model is ResUNet, a convolutional neural network architecture that combines residual connections with the U-Net structure. ResUNet is chosen for its ability to capture both local and global spatial features in medical images.

Key components of the model pipeline:

1. **Feature Extraction**: Convolutional and residual blocks identify complex patterns in retinal images, such as edges, vessels and lesions.  
2. **Classification Head**: Fully connected layers with a Softmax activation output probabilities for each class.  

---

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
