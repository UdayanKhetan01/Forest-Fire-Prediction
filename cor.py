from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
df = pd.read_csv("Algerian_forest_fires.csv")
print(df.columns)
# df.loc[df["area"] > 6, ["area"]] = 1


df['Classes'] = np.where(df["Classes"] == 'not fire', 0, 1)
print(df)
df.drop(df.columns[[0, 1, 2]], axis=1, inplace=True)
# head = list(df.columns)
# print(head)

# df.iloc[:, [5, 9, 10, 11]].values
# sc_x = StandardScaler()
# df_scaled = sc_x.fit_transform(df.to_numpy())
# df = pd.DataFrame(df_scaled, columns=[
#     'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area'])
# print(df)

# df= pd.DataFrame(df, columns=[
#   'sepal_length','sepal_width','petal_length','petal_width'])
# print(df)
plt.figure(figsize=(12, 10))
cor = df.corr()
print(cor)
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
