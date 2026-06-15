\# Brain Tumor Classification using VGG16 Transfer Learning



\## Overview



This project presents a Deep Learning-based medical image classification system designed to identify different types of brain tumors from MRI scans. The model leverages Transfer Learning using the VGG16 architecture pre-trained on ImageNet and fine-tuned for multi-class brain tumor classification.



The system classifies MRI images into four categories:



\* Glioma Tumor

\* Meningioma Tumor

\* Pituitary Tumor

\* No Tumor



The project demonstrates the application of Computer Vision and Deep Learning techniques in healthcare and medical diagnostics.



\---



\## Project Objectives



\* Develop an automated Brain MRI classification system.

\* Reduce manual diagnosis effort through AI-assisted prediction.

\* Apply Transfer Learning for improved performance and faster convergence.

\* Evaluate model performance using multiple classification metrics.



\---



\## Dataset



The project uses Brain MRI images organized into four classes:



```text

Training/

├── glioma

├── meningioma

├── pituitary

└── notumor



Testing/

├── glioma

├── meningioma

├── pituitary

└── notumor

```



Dataset images are resized to \*\*128 × 128 pixels\*\* before training.



\*\*Note:\*\* The dataset is not included in this repository because of size limitations.



\---



\## Data Preprocessing



The following preprocessing techniques were applied:



\* Image Resizing (128×128)

\* Pixel Normalization

\* Random Brightness Augmentation

\* Random Contrast Augmentation

\* Dataset Shuffling



These techniques help improve model generalization and reduce overfitting.



\---



\## Model Architecture



The model uses Transfer Learning with VGG16.



```text

Input Image (128×128×3)

&#x20;       │

&#x20;       ▼

Pre-trained VGG16

(ImageNet Weights)

&#x20;       │

&#x20;       ▼

Flatten Layer

&#x20;       │

&#x20;       ▼

Dropout (0.3)

&#x20;       │

&#x20;       ▼

Dense (128, ReLU)

&#x20;       │

&#x20;       ▼

Dropout (0.2)

&#x20;       │

&#x20;       ▼

Dense (4, Softmax)

```



\### Transfer Learning Strategy



\* All VGG16 layers are initially frozen.

\* The final layers are selectively unfrozen and fine-tuned.

\* ImageNet pre-trained weights are utilized to accelerate training.



\---



\## Technologies Used



\* Python

\* TensorFlow

\* Keras

\* VGG16 Transfer Learning

\* NumPy

\* Matplotlib

\* Seaborn

\* Scikit-Learn

\* PIL



\---



\## Training Configuration



| Parameter     | Value                           |

| ------------- | ------------------------------- |

| Image Size    | 128×128                         |

| Batch Size    | 20                              |

| Epochs        | 5                               |

| Optimizer     | Adam                            |

| Learning Rate | 0.0001                          |

| Loss Function | Sparse Categorical Crossentropy |



\---



\## Evaluation Metrics



The model is evaluated using:



\* Classification Report

\* Accuracy

\* Precision

\* Recall

\* F1 Score

\* Confusion Matrix

\* ROC Curve

\* Area Under Curve (AUC)



\---



\## Project Structure



```text

brain-tumor-classification-vgg16/

│

├── README.md

├── requirements.txt

├── LICENSE

│

├── notebooks/

│   └── brainCNN.ipynb

│

├── src/

│   ├── train.py

│   └── predict.py

│

├── screenshots/

│

├── dataset/

│   └── dataset\_link.txt

│

└── model/

```



\---



\## Running the Project



\### Clone Repository



```bash

git clone https://github.com/ayushthakur874485-arch/brain-tumor-classification-vgg16.git

```



\### Install Dependencies



```bash

pip install -r requirements.txt

```



\### Train Model



```bash

python src/train.py

```



\### Predict MRI Scan



```bash

python src/predict.py

```



\---



\## Future Improvements



\* Streamlit Web Application

\* Docker Containerization

\* REST API Integration

\* Cloud Deployment on AWS

\* CI/CD Pipeline Integration

\* Model Optimization and Hyperparameter Tuning



\---



\## Author



Ayush Thakur



GitHub:

https://github.com/ayushthakur874485-arch



LinkedIn:

https://linkedin.com/in/ayush-thakur-65423638a



