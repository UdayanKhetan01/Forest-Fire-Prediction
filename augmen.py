from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from genetic_selection import GeneticSelectionCV
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

dataset = pd.read_csv("Algerian_forest_fires.csv")
dataset['Classes'] = np.where(dataset["Classes"] == 'not fire   ', 0, 1)
print(round(dataset.Classes.value_counts(normalize=True), 4)*100)

x = dataset.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10, 11]].values

y = dataset.iloc[:, 13].values


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0)

sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)


# classifier = LogisticRegression(random_state=0)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)

# cm = confusion_matrix(y_test, y_pred)

# print("Confusion Matrix : \n", cm)

# print("Accuracy : ", accuracy_score(y_test, y_pred))


counter = Counter(y_train)
print('Before', counter)
smt = SMOTE()

X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
counter = Counter(y_train_sm)
print('After', counter)


# classifier = LogisticRegression(random_state=0)
# classifier.fit(X_train_sm, y_train_sm)
# y_pred = classifier.predict(X_test)

# cm = confusion_matrix(y_test, y_pred)

# print("Confusion Matrix : \n", cm)

# print("Accuracy : ", accuracy_score(y_test, y_pred))
