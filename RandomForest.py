#----------------------------------------------------------
# Feedback Moment: Module 2
# Implementation of a machine learning technique with the use of a framework.
#
# Date: 11-Sep-2023
# Author:
#           A01753176 Gilberto André García Gaytán
# Evaluation Metrics:
# R^2 (Coefficient of Determination): Represents the percentage of the total variability in the data explained by the model. A value close to 1 indicates a good fit of the model.
# MAE (Mean Absolute Error): Average of the absolute errors between predictions and actual values. Provides a direct sense of the error in the original units.
# MSE (Mean Squared Error): Average of the squared errors. It penalizes larger errors more.
# RMSE (Root Mean Squared Error): Square root of the MSE. Offers an interpretation of the error in original units and penalizes large errors.

# Ensure that this libraries are installed:
# pip install scikit-learn
# pip install pandas
# pip install matplotlib
# pip install numpy
# pip install Pillow
# pip install tkinter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load and preprocess data
def load_data():
    """
    The function `load_data` loads a CSV file, preprocesses the data by dropping certain columns,
    filling missing values, encoding categorical variables, and splits the data into training,
    validation, and test sets.
    :return: The function load_data() returns four variables: X_train, y_train, X_val, and y_val.
    """
    file_path = filedialog.askopenfilename(title="Select a CSV file", filetypes=[("CSV files", "*.csv")])
    data = pd.read_csv(file_path)
    data = data.drop(columns=['Unnamed: 0', 'name', 'full_name', 'place_of_birth', 'shirt_nr', 'player_agent', 'contract_expires', 'joined_club'])
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column].fillna(data[column].median(), inplace=True)

    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    X = data.drop('price', axis=1)
    y = data['price']
    # Splitting the data into train, temporary (for validation and test), and then validation and test sets.
    # 70% of the data goes to training (X_train and y_train).
    # 15% of the data goes to validation (X_val and y_val).
    # 15% of the data goes to testing (X_test and y_test).
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test

# Train the Random Forest model
def train_model(X_train, y_train):
    """
    The function `train_model` trains a random forest regression model using the given training data.
    :param X_train: X_train is the training data, which is a matrix or dataframe containing the features
    or independent variables used to train the model. Each row represents a sample or observation, and
    each column represents a feature
    :param y_train: The y_train parameter represents the target variable or the dependent variable in
    your dataset. It is the variable that you are trying to predict or model. In machine learning, it is
    common to split your dataset into features (X) and the target variable (y). The X_train parameter
    represents the features or
    :return: a trained Random Forest Regressor model.
    """
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

# Visualization with tkinter
def plot_data(rf_model, X_val, y_val):
    """
    The function `plot_data` takes in a random forest model, validation data, and validation labels, and
    plots a scatter plot comparing the actual prices with the predicted prices, along with displaying
    various metrics such as R^2, MAE, MSE, and RMSE.
    :param rf_model: The rf_model parameter is the trained random forest regression model that you want
    to use for prediction. It should be an instance of the RandomForestRegressor class from the
    scikit-learn library
    :param X_val: X_val is the validation set of input features. It is a matrix or dataframe containing
    the input features for which we want to make predictions. Each row represents a single data point,
    and each column represents a different feature
    :param y_val: The parameter `y_val` represents the actual target values for the validation set. It
    is a numpy array or pandas Series containing the true values of the target variable for each sample
    in the validation set
    """
    y_val_pred = rf_model.predict(X_val)
    # Calculate metrics
    mae = mean_absolute_error(y_val, y_val_pred)
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_val_pred)
    r2_percentage = round(r2 * 100, 2)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_val, y_val_pred, color='blue', alpha=0.6)
    ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=3)
    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    ax.set_title('Comparison of Actual vs. Predicted Prices')
    # Display metrics within the plot area
    metrics_text = f'R^2 (Prediction Accuracy): {r2_percentage}%\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=1, column=0, padx=20, pady=20)
    canvas.draw()

def on_load_button_click():
    """
    The function `on_load_button_click()` loads data, trains a random forest model, and plots the data.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    rf_model = train_model(X_train, y_train)
    plot_data(rf_model, X_val, y_val)

# The code `window = tk.Tk()` creates a new Tkinter window, which is the main window of the
# application.
window = tk.Tk()
window.title("Comparison of Actual vs. Predicted Prices")

# The code `window.geometry("800x600")` sets the size of the window to 800 pixels wide and 600 pixels
load_button = ttk.Button(window, text="Load CSV and Train Model", command=on_load_button_click)
load_button.grid(row=0, column=0, padx=20, pady=20)

# The code `window.mainloop()` starts the main event loop of the application, which is responsible for
window.mainloop()
