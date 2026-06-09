import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Create simple dataset
np.random.seed(0)
X = np.linspace(0, 10, 20).reshape(-1, 1)
print(X)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, 20)

# Different model complexities
degrees = [1, 3, 10]

plt.figure(figsize=(15,4))

for i, degree in enumerate(degrees):
    
    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )
    
    model.fit(X, y)
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    y_pred = model.predict(X_test)
    
    plt.subplot(1,3,i+1)
    plt.scatter(X, y)
    plt.plot(X_test, y_pred)
    plt.title(f"Degree {degree}")

plt.tight_layout()
plt.show()
