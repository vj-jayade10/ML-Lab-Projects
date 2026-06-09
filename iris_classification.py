#5th
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size = 0.3, random_state = 42
)



#create modles
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000), # Increased max_iter for convergence
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB()
}
for name, model in models.items():
  model.fit(x_train,y_train)
  print(x_test)
  y_pred = model.predict(x_test)
  print(y_pred)
  accuracy = accuracy_score(y_test,y_pred)
  print(f"{name} accuracy: {accuracy}",round(accuracy*100,2),"%")
  #25
