#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM

class OneClassSVMModel:
    def __init__(self, kernel='rbf', gamma=0.00005, nu=0.001):
        self.model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    
    def fit(self, X):
        """
        Fit the One-Class SVM model on the training data.
        """
        self.model.fit(X)
    
    def fit_transform(self, X):
        """
        Fit the model and transform the data using the decision function.
        """
        self.fit(X)
        return self.model.decision_function(X)
    
    def predict(self, Y):
        """
        Predict the decision function values for the given data.
        """
        decision_function = self.model.decision_function(Y)
        # Return the negative of decision function for anomaly detection use case
        return -decision_function
    
    def plot_decision_boundary(self, X, Y):
        """
        Plot the decision boundary and the data points.
        """
        # Create a grid to plot the decision boundary
        xx, yy = np.meshgrid(np.linspace(min(X[:, 0]) - 50, max(X[:, 0]) + 50, 500),
                             np.linspace(min(X[:, 1]) - 50, max(X[:, 1]) + 50, 500))

        Z = self.model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the data and the decision boundary
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', linestyles='dashed')
        plt.scatter(X[:, 0], X[:, 1], c='blue', s=20)
        plt.scatter(Y[:, 0], Y[:, 1], c='red', s=20)
        plt.xlabel('Current_residual')
        plt.ylabel('Voltage_residual')
        plt.title('ML Boundary for Voltage/Current Residuals')
        plt.show()

    def plot_decision_function_over_time(self, decision_function):
        """
        Plot the decision function values over time, with points above 0 in red.
        """
        above_zero = decision_function >= 0
        below_zero = decision_function < 0

        # Plot the decision function values over time
        plt.plot(np.arange(len(decision_function))[below_zero], decision_function[below_zero], 'o', markersize=5, label='Normal')
        plt.plot(np.arange(len(decision_function))[above_zero], decision_function[above_zero], 'ro', markersize=5, label='Abnormal')
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Decision Function Value')
        plt.title('ML Decision Function Over Time')
        plt.legend()
        plt.show()

#%%
if __name__ == "__main__":
    constant_res_train = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'Data', 'train3_1.csv'))
    constant_res_test = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'Data', 'train3_2.csv'))

    X = np.column_stack(
        (
            np.array(constant_res_train['Current_RES']),
            np.array(constant_res_train['Voltage_RES'])
            ))
    Y = np.column_stack(
        (
            np.array(constant_res_test['Current_RES']),
            np.array(constant_res_test['Voltage_RES'])
            ))

    # Create an instance of the class and use its methods
    model = OneClassSVMModel(kernel='rbf', gamma=0.00005, nu=0.001)
    model.fit(X)
    decision_function = model.predict(Y)
    model.plot_decision_boundary(X, Y)
    model.plot_decision_function_over_time(decision_function)

    decision_function = model.predict(Y)
    model.plot_decision_boundary(X, Y)
    model.plot_decision_function_over_time(decision_function)

# %%
