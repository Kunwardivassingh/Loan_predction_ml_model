import pandas as pd
import pickle

# Load the test dataset
df_test = pd.read_csv('test_loan_dataset.csv')

# Drop the Loan_ID column if present
if 'Loan_ID' in df_test.columns:
    df_test.drop(columns='Loan_ID', axis=1, inplace=True)

# Load the saved model
with open('loan_prediction_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Make predictions on the test dataset
y_pred = loaded_model.predict(df_test)

# Print predictions
y_pred_series = pd.Series(y_pred, name='Loan_Status_Prediction')
result = pd.concat([df_test, y_pred_series], axis=1)
print(result)
