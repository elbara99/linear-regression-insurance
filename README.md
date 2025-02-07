## Introduction
This project applies a linear regression model to predict medical insurance charges based on various features such as age, BMI, number of children, and more. The project uses a publicly available insurance dataset to train and evaluate the model.
## Features
- **Data Exploration (EDA):** Analyzes the dataset to understand relationships between features.
- **Data Transformation:** Converts categorical variables into numerical values using LabelEncoder.
- **Model Training:** Uses Linear Regression to build a predictive model.
- **Model Evaluation:** Assesses model performance with metrics such as MAE, MSE, RMSE, and R².
- **Visualization:** Includes plots for distribution, correlation, and residual analysis.

## Installation & Setup

### Prerequisites
- Python 3.x
- Required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

### Steps to run this project :
1. **Clone the Repository:**
  ```bash
   git clone https://github.com/your-username/insurance-linear-regression.git
  ```
2.Navigate to the Project Directory:
 ```bash
  cd insurance-linear-regression
```
3.Create and Activate a Virtual Environment:
 ```bash
  python -m venv env
# Activate on Windows:
env\Scripts\activate
# Or on macOS/Linux:
source env/bin/activate

```
4.Install Dependencies:
 ```bash
pip install -r requirements.txt
 ```
5.Run the Project:
 ```bash
python linear_regression.py

```
# Screenshots:
![coleration results](https://github.com/user-attachments/assets/a4fb83f1-3a51-4ec4-9fa6-e2583d9dddc4)
![distribution of medical charges](https://github.com/user-attachments/assets/b317fb21-9f59-486d-81c7-a81c6222b977)
![Correlation Heatmap](https://github.com/user-attachments/assets/08db57fe-0ad0-40af-ab94-df8bb2a5f162)
![resudual  histogram](https://github.com/user-attachments/assets/638eae30-6fc9-43b0-ab3c-38dd3a521e88)
![real values  vs produvtion](https://github.com/user-attachments/assets/2faf1429-ab51-4de7-a4e4-636b0fadba96)







