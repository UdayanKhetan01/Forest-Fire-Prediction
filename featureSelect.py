# Feature Importance with Extra Trees Classifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.feature_selection import SelectFromModel
# load data
dataset = pd.read_csv("Algerian_forest_fires.csv")
dataset['Classes'] = np.where(dataset["Classes"] == 'not fire   ', 0, 1)

x = dataset.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]].values
y = dataset.iloc[:, 13].values

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0)


sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

cols = [
    'Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
# y_train = pd.DataFrame(y_test, columns=['Classes'])
# y_test = pd.DataFrame(y_test, columns=['Classes'])


model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
# print(model.feature_importances_)
feature_scores = pd.Series(model.feature_importances_,
                           index=X_train.columns).sort_values(ascending=False)

print(feature_scores)

f_i = list(zip(cols, model.feature_importances_))
f_i.sort(key=lambda x: x[1])
plt.barh([x[0] for x in f_i], [x[1] for x in f_i])

plt.show()


# X_train = X_train.drop(['Ws'], axis=1)

# X_test = X_test.drop(['Ws'], axis=1)

# clf = RandomForestClassifier(n_estimators=100, random_state=0)


# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)


# print('Model accuracy score with Ws variable removed : {0:0.4f}'. format(
#     accuracy_score(y_test, y_pred)))


# sel = SelectFromModel(RandomForestClassifier(n_estimators=100))
# sel.fit(X_train, y_train)
# print(sel.get_support())
# selected_feat = X_train.columns[(sel.get_support())]
# print(selected_feat)
