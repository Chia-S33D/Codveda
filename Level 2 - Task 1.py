# 31 May 2026
# Task 1: Regression analysis
# Needed: Python, Scikit learn, and matplotlib

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# 1. LOAD DATA
# The house_Prediction_Data_Set.csv dataset has no headers and is sepe=arated by whitespaces.
# Assign column names manually: This is the classic Boston Housing

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

df = pd.read_csv(r"C:\Users\User\OneDrive\UCT\Year 3\Codveda\Data Set For Task-2026\Data Set For Task\4) house Prediction Data Set.csv", header = None, sep = r'\s+', names = column_names)

print(df.head())
print(df.info())
print(df.describe())


# 2. DEFINE FEATURES AND TARGET 
# x = all columns except MEDV (the features/inputs)
# y = MEDV (the target we want to predict)

x = df.drop(columns = ['MEDV'])
y = df['MEDV']


# 3. SPLIT INTO TRAIN AND TEST SETS
# 80% of data used for training, 20% for testing
# random_state = 42 ensures we get the same split every time we run the script

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(x_train)}")
print(f"Testing samples: {len(x_test)}")


# 4. LINEAR REGRESSION 

lm = LinearRegression()
lm.fit(x_train, y_train) # so the model learns from the training data


# 5. MAKING PREDICTIONS ^^

y_pred  = lm.predict(x_test)


# 6. EVALUATING THE MODEL

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f"MEAN SQUARED ERROR (MSE: {mse:.2f})")
print(f"R-squared: {r2:.4f}")


# 7. INTERPRET THE FIT COEFFICIENTS

df_coeff = pd.DataFrame({
           'Feature': x.columns,
           'Coefficient': lm.coef_
}).sort_values('Coefficient', ascending= False)

print(df_coeff.to_string(index=False))
print(f"\nIntercept: {lm.intercept_: .4f}")


# 8. VISUALIZATIONS USING PLOTS - To see how well our model predicts the house prices

fig, axes = plt.subplots(1, 2, figsize= (14,5))

# Plot 1: Actual vs Predicted prices
axes[0].scatter(y_test, y_pred, alpha = 0.6, color= 'steelblue', edgecolors = 'k', linewidths = 0.4)
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', linewidth = 1.5, label = "Actual Price")
axes[0].set_xlabel('Actual Price (MEDV)')
axes[0].set_ylabel('Predicted Price (MEDV)')
axes[0].set_title('Actual vs Predicted House Prices')
axes[0].legend()

# Plot 2: The Bar chart for the feature coefficients
colours = ['rebeccapurple' if c > 0 else 'indianred' for c in df_coeff['Coefficient']]
axes[1].barh(df_coeff['Feature'], df_coeff['Coefficient'], color = colours)
axes[1].set_xlabel('Coefficents')
axes[1].set_title('Feature Coefficients\n(rebeccapurple = positive effect, indianred = negative effect)')
axes[1].axvline(0, color = 'black', linewidth = 0.8)

plt.tight_layout()
plt.show()