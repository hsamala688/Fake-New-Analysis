import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Loading and Labeling the Data
fake_news_df = pd.read_csv('Fake.csv')
real_news_df = pd.read_csv('True.csv')

fake_news_df['label'] = 0 #This is Fake
real_news_df['label'] = 1 #This is Real

df = pd.concat([fake_news_df, real_news_df]).sample(frac=1).reset_index(drop=True)

#Selecting the Features and Target
X = df['text']
y = df['label']

#Training and Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

#Vectorizing tfidf
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

#The LR model
lr_model = LinearRegression()
lr_model.fit(X_train_tfidf, y_train)

#Predictions and our Evaluation
y_pred = lr_model.predict(X_test_tfidf)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

#Distribution of the predicted Values
plt.figure(figsize=(8,5))
plt.hist(y_pred, bins=30)
plt.title("Distribution of Linear Regression Predictions")
plt.xlabel("Predicted Value (0 = Fake, 1 = Real)")
plt.ylabel("Frequency")
plt.savefig("Distribution_of_Predicted_Values")
plt.show()

#The actual vs predicted scatter Plot
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.title("Actual vs Predicted (Linear Regression)")
plt.xlabel("Actual Label")
plt.ylabel("Predicted Value")
plt.savefig("Actual_vs_Predicted_Scatter_Plot")
plt.show()

#The Residual Plot
residuals = y_test - y_pred

plt.figure(figsize=(8,5))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(0, color='black', linestyle='--')
plt.title("Residual Plot")
plt.xlabel("Predicted Value")
plt.ylabel("Residual (Actual - Predicted)")
plt.savefig("Residuals_Plot")
plt.show()

#Linear Regression done by Kavin Ramesh, Issue with pushing to GitHub