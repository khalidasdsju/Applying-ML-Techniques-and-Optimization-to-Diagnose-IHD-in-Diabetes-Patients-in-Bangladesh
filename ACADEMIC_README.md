# Applying Machine Learning Techniques and Optimization to Diagnose Ischemic Heart Disease (IHD) in Diabetes Patients in Bangladesh

## Abstract

Ischemic heart disease (IHD) is a predominant cause of morbidity and mortality, especially among diabetic individuals in Bangladesh. Early detection of ischemic heart disease is essential for appropriate intervention and improved patient outcomes. Machine learning (ML) methodologies have demonstrated potential in improving diagnostic precision for ischemic heart disease (IHD). This research utilizes machine learning methodologies and optimization techniques to enhance the identification of ischemic heart disease in diabetic patients in Bangladesh. The main goal of this research is to examine the efficacy of machine learning techniques, encompassing algorithm optimization and ensemble learning, for identifying ischemic heart disease in diabetic patients via a cross-sectional study conducted in 2024.

## Introduction

Cardiovascular diseases, particularly ischemic heart disease (IHD), represent a significant global health burden, with a disproportionate impact on developing nations like Bangladesh. The co-occurrence of diabetes mellitus further exacerbates the risk and progression of IHD, creating a complex clinical scenario that demands accurate and timely diagnostic approaches. Traditional diagnostic methods for IHD in diabetic patients often face challenges related to sensitivity, specificity, and accessibility, particularly in resource-constrained settings.

This research addresses these challenges by leveraging advanced machine learning techniques to develop robust predictive models for IHD diagnosis in diabetic patients. By analyzing a comprehensive dataset collected from a cross-sectional study in Bangladesh, we aim to identify optimal algorithmic approaches and feature combinations that maximize diagnostic accuracy while maintaining clinical interpretability.

## Methodology

### Data Collection and Preprocessing

A dataset comprising clinical, demographic, and laboratory data from diabetic patients was collected through a cross-sectional study conducted in Bangladesh in 2024. The dataset includes the following key variables:

- Demographic information (age, sex, occupation, education level, economic status)
- Anthropometric measurements (height, weight, BMI)
- Clinical parameters (blood pressure, random blood sugar)
- Medical history (smoking status, hypertension, dyslipidemia, stroke)
- Target variable: Ischemic Heart Disease (IHD) status

Data preprocessing involved:
1. Handling missing values through median imputation
2. Outlier detection and treatment using IQR method and winsorization
3. Feature engineering to derive clinically relevant variables
4. Categorical variable encoding
5. Feature scaling and normalization

### Machine Learning Algorithms

We analyzed the dataset utilizing fourteen distinct machine learning algorithms:

1. Logistic Regression (LR)
2. k-Nearest Neighbors (kNN)
3. Naive Bayes (NB)
4. Decision Tree (DT)
5. Support Vector Machine (SVM)
6. Ridge Classifier (RC)
7. Random Forest (RF)
8. Quadratic Discriminant Analysis (QDA)
9. AdaBoost
10. Gradient Boosting (GB)
11. Linear Discriminant Analysis (LDA)
12. Extra Trees Classifier (ETC)
13. Classifier Chain (CC)
14. Decision Forest (DF)

### Model Optimization

Five-fold cross-validation was utilized for hyperparameter adjustment to enhance the models' predictive accuracy. Grid search optimization was employed to identify optimal hyperparameter configurations for each algorithm. The effectiveness of various models was evaluated based on multiple metrics:

- Accuracy
- Precision
- Recall
- F1 score
- ROC AUC
- Log loss

Additionally, we implemented ensemble learning techniques, specifically soft voting, to potentially improve predictive performance beyond individual algorithms.

## Results

The evaluation metrics indicate that Gradient Boosting is the most effective model, with an accuracy of 0.910 and a ROC AUC of 0.9694. It consistently outperforms other models in these areas, indicating that it is the most reliable model for classification tasks where predicted accuracy and class distinction are critical. 

However, in terms of log loss, Random Forest demonstrates higher performance with a lower value of 0.2493, signifying its enhanced reliability in probabilistic predictions. Nevertheless, Gradient Boosting's exceptional performance in accuracy and ROC AUC makes it the preferable choice for most applications.

The feature importance analysis revealed that the most significant predictors for IHD in diabetic patients include:
- Age
- Systolic blood pressure
- Random blood sugar levels
- Presence of hypertension
- Smoking status

## Discussion

The superior performance of Gradient Boosting can be attributed to its ability to sequentially correct errors and handle complex non-linear relationships between predictors and the outcome variable. This is particularly relevant in the context of IHD diagnosis, where multiple risk factors interact in complex ways to influence disease development and progression.

The identification of key predictive features aligns with established clinical knowledge regarding IHD risk factors, providing validation for our machine learning approach. Moreover, the high accuracy and ROC AUC values suggest that machine learning models can effectively augment clinical decision-making in resource-constrained settings where access to advanced diagnostic modalities may be limited.

The slightly better performance of Random Forest in terms of log loss indicates its potential utility in scenarios where calibrated probability estimates are more important than binary classification outcomes. This could be relevant in risk stratification applications where the degree of certainty is as important as the classification itself.

## Conclusion

This study highlights the potential of Machine Learning Techniques and Optimization, with Gradient Boosting being the most effective model for overall classification performance. Random Forest is a viable alternative for reducing log loss. Soft Voting Ensemble, while successful, falls short of the top models. 

The study also suggests that integrating GridSearchCV, five-fold cross-validation, and soft voting ensemble classifiers can help in early diagnosis and treatment planning for diabetes patients with ischemic heart disease. These findings have significant implications for clinical practice in Bangladesh and similar settings, where early and accurate diagnosis of IHD in diabetic patients can facilitate timely intervention and improved outcomes.

## Future Directions

Future research should focus on:
1. External validation of the models in diverse patient populations
2. Incorporation of additional biomarkers and imaging data to enhance predictive performance
3. Development of interpretable AI approaches that can provide actionable insights for clinicians
4. Implementation studies to assess the real-world impact of ML-based diagnostic tools in clinical settings
5. Longitudinal studies to evaluate the models' ability to predict IHD development in diabetic patients over time

## Acknowledgments

We acknowledge the support of [relevant institutions] and the contributions of all study participants. This research was conducted in accordance with ethical guidelines and received approval from [relevant ethics committee].

## References

[List of relevant references in academic format]
