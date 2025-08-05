# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = pd.read_csv('/kaggle/input/titanic-dataset/Titanic-Dataset.csv')  

print("setup complete")


# Basic info
print("DataFrame Info:")
print(df.info())

# Descriptive statistics for all columns
print("\nDescriptive Statistics:")
print(df.describe(include='all'))

# Median of numeric columns
print("\nMedian values:")
print(df.median(numeric_only=True))


# Get only numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Histograms for all numeric columns
df[numeric_cols].hist(figsize=(10, 8), bins=20)
plt.suptitle("Histograms of Numeric Features", fontsize=16)
plt.tight_layout()
plt.show()

# Boxplots for all numeric columns
plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_cols):
    plt.subplot(2, len(numeric_cols)//2 + 1, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()


# Correlation matrix (numeric only)
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Pairplot (takes time)
sns.pairplot(df[numeric_cols].dropna())
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.show()

# Survival count
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Survival rate by gender
sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate by Sex")
plt.show()

# Survival rate by class
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()

# Age distribution by survival
sns.histplot(data=df, x='Age', hue='Survived', multiple='stack')
plt.title("Age Distribution by Survival")
plt.show()

# Embarked vs Survival
sns.barplot(x='Embarked', y='Survived', data=df)
plt.title("Survival Rate by Embarked Port")
plt.show()

