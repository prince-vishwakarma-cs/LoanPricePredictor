# Insurance Price Prediction using Gradient Boosting

## Overview
This project aims to build a predictive model for estimating insurance prices based on various risk factors. By leveraging **Gradient Boosting Regression**, the model achieves a high level of accuracy and generalization across diverse datasets. The model has been optimized using **Grid Search CV** to ensure robust performance.

## Features
- **Accurate Price Prediction**: Utilizes Gradient Boosting to model complex relationships between features and insurance pricing.
- **Feature Engineering & Transformation**: Prepares data for optimal model performance.
- **Hyperparameter Tuning**: Employs Grid Search to find the best model parameters.
- **Evaluation Metrics**: Assesses model performance using R² score to ensure reliability.
- **Scalable & Reproducible**: The pipeline is structured for easy modifications and scalability.

## Tech Stack
- **Programming Language**: Python
- **Libraries**: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning Model**: Gradient Boosting Regression

## Dataset
The dataset contains multiple features relevant to insurance pricing, including:
- **Age**: The age of the insured individual.
- **BMI**: Body mass index, an indicator of health risk.
- **Smoking Status**: A key factor in determining insurance cost.
- **Region**: The geographical location affecting pricing.
- **Number of Dependents**: Family structure influencing policy rates.

## Installation & Setup
To run this project locally, follow these steps:

### Prerequisites
Ensure you have Python installed (version 3.7+ recommended). Install the required dependencies using:
```bash
pip install -r requirements.txt
```

### Clone the Repository
```bash
git clone https://github.com/yourusername/insurance-price-prediction.git
cd insurance-price-prediction
```

## Model Training & Evaluation
### 1. Data Preprocessing
- Handle missing values and outliers.
- Encode categorical variables.
- Normalize and scale features.

### 2. Model Training
```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# Define model
model = GradientBoostingRegressor()

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)
```

### 3. Model Evaluation
```python
from sklearn.metrics import r2_score

print("Test accuracy : ", r2_score(y_test, grid_search.predict(X_test)))
print("Train accuracy : ", r2_score(y_train, grid_search.predict(X_train)))
```
- **Achieved R² score**: ~0.88 on test data, indicating strong predictive performance.

## Results & Insights
- The **Gradient Boosting model** outperformed baseline regression models.
- **Feature Importance Analysis** showed that smoking status, BMI, and age had the most significant impact on insurance pricing.
- **Hyperparameter tuning** improved generalization and minimized overfitting.

## Future Enhancements
- Incorporate **deep learning techniques** for further accuracy improvements.
- Deploy as a **web application** for real-world usability.
- Integrate **real-time data pipelines** for continuous model updates.

## Contributing
Feel free to fork this repository and submit pull requests for improvements. Contributions are welcome!

## License
This project is licensed under the MIT License.

## Contact
For any queries, reach out via:
- **Email**: your.email@example.com
- **LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile)
- **GitHub**: [Your Repository](https://github.com/yourusername)

