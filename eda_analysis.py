import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("Customer_Sales_EDA_Dataset.csv")

# Display First Rows
print(df.head())

# Dataset Information
print("\nDataset Info:")
print(df.info())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Average Purchase Amount by Category
avg_purchase = df.groupby("Product_Category")["Purchase_Amount"].mean()

print("\nAverage Purchase by Category:")
print(avg_purchase)

# Bar Chart
avg_purchase.plot(kind='bar')

plt.title("Average Purchase Amount by Category")
plt.xlabel("Product Category")
plt.ylabel("Average Purchase Amount")
plt.show()

# Customer Rating Distribution
plt.hist(df["Customer_Rating"])

plt.title("Customer Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# Age vs Purchase Amount
plt.scatter(df["Age"], df["Purchase_Amount"])

plt.title("Age vs Purchase Amount")
plt.xlabel("Age")
plt.ylabel("Purchase Amount")
plt.show()

# Correlation Matrix
correlation = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(correlation)