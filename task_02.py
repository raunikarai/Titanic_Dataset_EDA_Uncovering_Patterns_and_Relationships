import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

titanic_df = pd.read_csv("titanic.csv")
print(titanic_df.head())
print(titanic_df.isnull().sum())
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
titanic_df.drop('Cabin', axis=1, inplace=True)
titanic_df.dropna(subset=['Embarked'], inplace=True)


sns.countplot(x='Pclass', data=titanic_df)
plt.title('Passenger Class Distribution')
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=titanic_df)
plt.title('Survival Count based on Passenger Class')
plt.show()


sns.countplot(x='Sex', data=titanic_df)
plt.title('Gender Distribution')
plt.show()


sns.countplot(x='Sex', hue='Survived', data=titanic_df)
plt.title('Survival Count based on Gender')
plt.show()


sns.histplot(x='Age', data=titanic_df, bins=20, kde=True)
plt.title('Age Distribution of Passengers')
plt.show()


sns.histplot(x='Age', hue='Survived', data=titanic_df, bins=20, kde=True)
plt.title('Survival Distribution based on Age')
plt.show()


sns.histplot(x='Fare', data=titanic_df, bins=30, kde=True)
plt.title('Fare Distribution')
plt.show()


sns.histplot(x='Fare', hue='Survived', data=titanic_df, bins=30, kde=True)
plt.title('Survival Distribution based on Fare')
plt.show()


numeric_columns = titanic_df.select_dtypes(include=[np.number]).columns


corr_matrix = titanic_df[numeric_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()