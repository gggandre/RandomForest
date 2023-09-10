# RandomForest

## Link tutorial to run the code: https://github.com/gggandre/RandomForest/assets/84719490/6307bc8f-7d9c-4b5c-839a-ea7c9966c048

## Step by step Random Forest Regressor to predict player prices based on various features. 

### Loading and Preprocessing Data:

**Function:** ```load_data()```
**Purpose:** This function loads a CSV file, preprocesses the data, and splits it into training, validation, and test sets.
**Steps:**
1. The user is prompted to select a CSV file using a file dialog.
2. The data is read from the CSV file.
3. Certain columns (like name, full_name, etc.) are dropped as they're not needed for prediction.

**Missing values in columns are filled:**
- For categorical columns (object dtype), missing values are filled with the mode.
- For numerical columns, missing values are filled with the median.
- Categorical columns are encoded using LabelEncoder to convert them to numerical values.
- The data is split into features (X) and the target (y - which is 'price' in this case).
- The data is then split into training, validation, and test sets. 70% for training, 15% for validation, and 15% for testing.
        
### Training the Random Forest Model:

**Function:** train_model(X_train, y_train)
**Purpose: **This function trains a Random Forest regression model using the provided training data.
**Steps:**
1. A Random Forest Regressor model is initialized.
2. The model is trained using the training data (X_train and y_train).

### Visualization and Model Evaluation:

**Function:** plot_data(rf_model, X_val, y_val)
**Purpose:** This function visualizes the comparison between actual and predicted prices and displays evaluation metrics.
**Steps:**
1. Predictions are made on the validation data (X_val) using the trained Random Forest model.
2. Various metrics are calculated to evaluate the model's performance:
- R^2 (Coefficient of Determination): Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
- MAE (Mean Absolute Error): The average of the absolute differences between predictions and actual values.
- MSE (Mean Squared Error): The average of the squared differences between predictions and actual values.
 - RMSE (Root Mean Squared Error): The square root of the MSE.
3. A scatter plot is created comparing the actual prices (y_val) with the predicted prices.
- The metrics (R^2, MAE, MSE, RMSE) are displayed within the plot area.

### Tkinter GUI Interface:

- The code uses Tkinter to create a simple graphical user interface.
- The main window is created and titled "Comparison of Actual vs. Predicted Prices".
- A button labeled "Load CSV and Train Model" is provided. When clicked:
- The on_load_button_click() function is triggered.
- This function in turn calls load_data(), train_model(), and plot_data() in sequence to load the data, train the model, and visualize the results.
- The main event loop of the application is started with window.mainloop(), which keeps the application running.

# Run code

1. Clone the repo or download the .zip
![image](https://github.com/gggandre/RandomForest/assets/84719490/8eeb21b0-c039-4bb8-9ba4-04dffb530aa5)

2. Open the folder in your local environment (make sure you have Python installed, along with the libraries numpy, pandas, tkinter,sklearn,matloptlib,Pillow)
![image](https://github.com/gggandre/RandomForest/assets/84719490/153973cd-3472-4724-b63c-0b59d46e2c56)
- Note: Note: If the libraries are not installed, use ```pip install pandas numpy tkinter matplotlib Pillow scikit-learn```

3. Run the code with code runner or terminal
![image](https://github.com/gggandre/Kmeans/assets/84719490/92eecb31-649d-417b-8fdc-59dc7d533675)
![image](https://github.com/gggandre/RandomForest/assets/84719490/251528a5-1238-415e-85b4-05baf068a9bb)

4. Click in the "Load CSV and Train Model" window
![image](https://github.com/gggandre/RandomForest/assets/84719490/1748cd30-cda5-4f5d-bbb6-3375c23fced3)

5. Select the csv file (top5_leagues_player.csv)
![image](https://github.com/gggandre/RandomForest/assets/84719490/8dfbbdd5-d151-41c3-a16b-5679bab65bd9)

6. Now you can see the Comparision of Actual prices vs predicted prices with the validation metrics
![image](https://github.com/gggandre/RandomForest/assets/84719490/e1db94bd-4ebf-484d-8784-0801532899a3)
