from setuptools import setup, find_packages

setup(
    name="ihd_diagnosis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "scipy",
        "joblib",
        "flask",
        "pyspss",
        "imblearn",
        "lightgbm",
        "statsmodels",
    ],
    author="Khalid",
    author_email="example@example.com",
    description="Machine Learning for IHD Diagnosis in Diabetes Patients",
    keywords="machine learning, healthcare, diabetes, heart disease",
    python_requires=">=3.8",
)
