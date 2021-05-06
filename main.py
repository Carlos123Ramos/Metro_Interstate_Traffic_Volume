
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd

dataset = pd.read_csv('Metro_Interstate_Traffic_Volume.csv', delimiter=",")

X = dataset.iloc[:,0:8]
y = dataset.iloc[:,8:9].values

X, y = make_classification(n_samples=200, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

regr = MLPClassifier(random_state=1,activation = 'relu', max_iter=300).fit(X_train, y_train)

performance = regr.predict([[34,34,23,34,34,23,34,34,23,34,34,23,34,34,23,34,34,23,3,23]])
print (performance[0])

print(regr.score(X_test, y_test))

