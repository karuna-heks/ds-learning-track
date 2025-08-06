import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

input_dir = 'data/raw'
input_file = 'Titanic-Dataset.csv'
results_dir = 'data/processed'

df = pd.read_csv(f'{input_dir}/{input_file}')

df = df[~(df['Embarked'].isna())]
df['Age'] = df['Age'].fillna(df['Age'].mean())
df.drop(columns=['Cabin'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Embarked']], df['Survived'], test_size=0.2, random_state=42)

X_train.to_csv(f'{results_dir}/X_train.csv', index=False)
X_test.to_csv(f'{results_dir}/X_test.csv', index=False)
y_train.to_csv(f'{results_dir}/y_train.csv', index=False)
y_test.to_csv(f'{results_dir}/y_test.csv', index=False)