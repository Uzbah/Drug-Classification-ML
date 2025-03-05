# Drug Classification Project

## Overview

This project focuses on classifying drugs based on various patient attributes using machine learning techniques. The dataset contains information about patients, such as age, sex, blood pressure (BP), cholesterol levels, and the type of drug prescribed. The goal is to build a classification model that can accurately predict the type of drug a patient should be prescribed based on their characteristics.

## Dataset

The dataset used in this project is named `drug_classification.csv`. It contains the following features:

- **Age**: Age of the patient
- **Sex**: Gender of the patient (Male/Female)
- **BP**: Blood pressure level (High/Normal/Low)
- **Cholesterol**: Cholesterol level (High/Normal)
- **Na_to_K**: Sodium to Potassium ratio in the blood
- **Drug_Type**: The type of drug prescribed (Target variable)

## Project Structure

The project is structured as follows:

- **`drugclassification.py`**: The main Python script containing the code for data preprocessing, model training, hyperparameter tuning, and evaluation.
- **`drug_classification.csv`**: The dataset used for training and testing the models.
- **`README.md`**: This file, providing an overview of the project and instructions for running it.

## Steps Performed

1. **Data Loading and Exploration**:
   - The dataset was loaded and explored to understand its structure and check for missing or duplicate values.
   - Basic statistics and data types were examined to ensure data quality.

2. **Data Preprocessing**:
   - The dataset was split into features (`X`) and the target variable (`y`).
   - Categorical variables (Sex, BP, Cholesterol) were converted into numerical format using one-hot encoding.
   - The data was split into training and testing sets with an 80-20 ratio.

3. **Model Training and Evaluation**:
   - Three classification algorithms were applied:
     - **K-Nearest Neighbors (KNN)**
     - **Random Forest (RF)**
     - **Logistic Regression (LR)**
   - Each model was trained on the training data and evaluated on the test data using accuracy and confusion matrices.
   - Classification reports were generated to assess the performance of each model.

4. **Hyperparameter Tuning**:
   - Grid Search Cross-Validation was used to find the best hyperparameters for each model:
     - **KNN**: Tuned for the number of neighbors.
     - **Random Forest**: Tuned for the number of estimators, maximum depth, and minimum samples split.
     - **Logistic Regression**: Tuned for regularization strength (`C`) and solver type.
   - The tuned models were evaluated again to check for improvements in performance.

5. **Results and Observations**:
   - **Random Forest** and **Logistic Regression** achieved perfect accuracy (100%) on the test data, indicating they are well-suited for this classification task.
   - **KNN** struggled with imbalanced classes, particularly for drugs with fewer samples (e.g., drugC and drugB), resulting in lower accuracy even after tuning.

## Key Findings

- **Best Performing Models**: Random Forest and Logistic Regression performed exceptionally well, achieving 100% accuracy on the test data.
- **KNN Limitations**: KNN's performance was hindered by class imbalance, making it less effective for this dataset.
- **Class Imbalance Impact**: Classes with fewer samples (e.g., drugC and drugB) were harder to predict, especially for KNN.

## Requirements

To run this project, you will need the following Python libraries:

- `pandas`
- `scikit-learn`
- `numpy`

You can install the required libraries using pip:

```bash
pip install pandas scikit-learn numpy
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/drug-classification.git
cd drug-classification
```

2. Ensure you have the dataset `drug_classification.csv` in the same directory as the script.

3. Run the script using Python:

```bash
python drugclassification.py
```

## Results

After running the script, you will see the following outputs:

- Accuracy scores for KNN, Random Forest, and Logistic Regression models.
- Confusion matrices for each model.
- Classification reports showing precision, recall, and F1-score for each class.
- Best hyperparameters found during tuning for each model.

## Conclusion

This project demonstrates the effectiveness of different machine learning algorithms for drug classification. Random Forest and Logistic Regression emerged as the best models for this task, while KNN struggled due to class imbalance. Future work could involve addressing class imbalance through techniques like oversampling or using more advanced models to improve KNN's performance.

---

**Note**: This project was developed in a Google Colab environment, but it can be adapted to run locally with minor modifications. For any questions or issues, feel free to open an issue on GitHub.
