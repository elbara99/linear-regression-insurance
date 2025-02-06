import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('insurance.csv')

print("head:")
print(data.head())

print("info:")
print(data.info())

print("describe:")
print(data.describe())

plt.figure(figsize=(8, 6))
sns.histplot(data['charges'], kde=True, bins=30)
plt.title("Distribution of Medical Charges")
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.show()

sns.pairplot(data)
plt.show()

from sklearn.preprocessing import LabelEncoder

data_encoded = data.copy()
le = LabelEncoder()
data_encoded['sex'] = le.fit_transform(data_encoded['sex'])
data_encoded['smoker'] = le.fit_transform(data_encoded['smoker'])
data_encoded['region'] = le.fit_transform(data_encoded['region'])

print("Data after transforming categorical variables:")
print(data_encoded.head())

plt.figure(figsize=(10, 8))
sns.heatmap(data_encoded.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

X = data_encoded.drop('charges', axis=1)
y = data_encoded['charges']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

print("\nIntercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

predictions = model.predict(X_test)

df_compare = pd.DataFrame({'Real_Values': y_test, 'Predictions': predictions})
print("combining ::")
print(df_compare.head())

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=predictions)
plt.xlabel('Real Values')
plt.ylabel('Predictions')
plt.title('Real Values vs Predictions')
plt.show()

residuals = y_test - predictions

plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.title('Residual Histogram')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("Rating:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R^2 Score:", r2)
