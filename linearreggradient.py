# %%
import pandas as pd
import numpy as np


# %%
class LinearRegssionGradientModel:
    def __init__(self, learning_rate = 0.01, epochs = 1000):
        self.learning_rate = learning_rate
        self.weight = None
        self.bias = None
        self.epochs = epochs
        self.cost_history = []

    def fit(self, X, y): 

        #initialization of the parameters

        num_samples, num_features = X.shape
        self.weight = np.zeros(num_features)
        self.bias = 0.0

        #gradient descend loop
        for epochs in range(self.epochs):
            #make the prediction y hat = Xw + b
            y_predict = np.dot(X, self.weights) + self.bias
            cost_function = (1/2 * (num_samples)) * np.sum( (y_predict - y) ** 2)
            self.cost_history.append(cost_function)

            #Calculating the gradients
            dw = (1/ num_samples) * np.dot(X.T, (y_predict - y))
            db = (1/num_samples) * np.sum(y_predict - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        def predict(self, X):
            return np.dot(X, self.weights) + self.bias




# %%
#example for usinng it
if __name__ == "__main__":
    X = 2 * np.random.rand(100, 1)
    y = 5 + 2 * X.squeez() + np.random.randn(100)
    model = LinearRegssionGradientModel(learning_rate=0.01, epochs = 10000)
    model.fit(X, y)
    model.predict(X)


