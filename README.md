# Applying Machine Learning Techniques and Optimization to Diagnose IHD in Diabetes Patients

## Project Overview

This project applies machine learning techniques and optimization to diagnose Ischemic Heart Disease (IHD) in diabetes patients in Bangladesh. The project uses data from a cross-sectional study conducted in 2024.

## Abstract

**Background**: Ischemic heart disease (IHD) is a predominant cause of morbidity and mortality, especially among diabetic individuals in Bangladesh. Early detection of ischemic heart disease is essential for appropriate intervention and improved patient outcomes. Machine learning (ML) methodologies have demonstrated potential in improving diagnostic precision for ischemic heart disease (IHD). This project utilizes machine learning methodologies and optimization techniques to enhance the identification of ischemic heart disease in diabetic patients in Bangladesh.

**Methods**: A dataset comprising clinical, demographic, and laboratory data from diabetic patients was analyzed utilizing fourteen distinct machine learning algorithms: Logistic Regression (LR), k-Nearest Neighbors (kNN), Naive Bayes (NB), Decision Tree (DT), Support Vector Machine (SVM), Ridge Classifier (RC), Random Forest (RF), Quadratic Discriminant Analysis (QDA), AdaBoost, Gradient Boosting (GB), Linear Discriminant Analysis (LDA), Extra Trees Classifier (ETC), Classifier Chain (CC), and Decision Forest (DF). Five-fold cross-validation was utilized for hyperparameter adjustment to enhance the model's predictive accuracy.

**Results**: The evaluation metrics indicate that Gradient Boosting is the most effective model, with an accuracy of 0.910 and a ROC AUC of 0.9694. It consistently outperforms other models in these areas, indicating that it is the most reliable model for classification tasks where predicted accuracy and class distinction are critical.

**Conclusions**: The study highlights the potential of Machine Learning Techniques and Optimization, with Gradient Boosting being the most effective model for overall classification performance. Random Forest is a viable alternative for reducing log loss. The study also suggests that integrating GridSearchCV, five-fold cross-validation, and soft voting ensemble classifiers can help in early diagnosis and treatment planning for diabetes patients with ischemic heart disease.

## Project Structure

```
.
├── data/                  # Data directory
├── models/                # Saved models
├── notebooks/             # Jupyter notebooks
├── src/                   # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # Model training and evaluation
│   ├── utils/             # Utility functions
│   └── app/               # Web application
├── tests/                 # Unit tests
├── requirements.txt       # Dependencies
├── setup.py               # Package installation
└── README.md              # Project documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ihd-diagnosis.git
cd ihd-diagnosis

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

### Data Processing

```python
from src.data.preprocessing import preprocess_data

# Load and preprocess data
data = preprocess_data('data/ASDS_Study_Data.sav')
```

### Model Training

```python
from src.models.train_model import train_model

# Train a model
model = train_model(data)
```

### Model Evaluation

```python
from src.models.evaluate_model import evaluate_model

# Evaluate the model
metrics = evaluate_model(model, test_data)
```

### Web Application

```bash
# Run the web application
python src/app/app.py
```
