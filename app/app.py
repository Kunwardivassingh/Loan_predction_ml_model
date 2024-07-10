import flet as ft
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score,classification_report 
from sklearn.model_selection import train_test_split


data = None  # To hold the uploaded dataset
selected_model = None  # To hold the selected classifier model
model = None  # To hold the built model


# Function to preprocess data and build a model
def build_model(data, model_type):
    x = data.drop('Loan_Status', axis=1)
    y = data['Loan_Status']

    numeric_features = x.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = x.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)])

    if model_type == 'Random Forest':
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=200))
        ])
    elif model_type == 'Decision Tree':
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier())
        ])
    elif model_type == 'Gradient boost':
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier())
            ])
    # Add more elif conditions for other models as needed

    model.fit(x, y)
    return model

def main(page: ft.Page):
    page.title = "Loan Prediction App"
    page.scroll = "auto"
    
    # Define the upload interface
    def upload_interface():
        def on_file_upload(e: ft.FilePickerResultEvent):
            global data
            if e.files:
                file_path = e.files[0].path
                data = pd.read_csv(file_path)
                data_head.value = f"HEAD:\n{data.head().to_string()}"
                data_tail.value = f"TAIL:\n{data.tail().to_string()}"
                data_describe.value = f"DESCRIBE:\n{data.describe().to_string()}"
                num_cols = data.select_dtypes(include=['float64', 'int64']).columns
                data[num_cols] = data[num_cols].fillna(data[num_cols].median())
                # Filling categorical columns with mode (if any)
                catg_cols = data.select_dtypes(include=['object']).columns
                data[catg_cols] = data[catg_cols].fillna(data[catg_cols].mode().iloc[0])
                missing_values.value = f"MISSING VALUES:\n{data.isnull().sum().to_string()}"
                missing_values.value = f"MISSING VALUES:\n{data.isnull().sum().to_string()}"
                
                # Identify outliers in numeric columns
                numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
                outliers_data = data[(data[numeric_cols] - data[numeric_cols].mean()).abs() > 3 * data[numeric_cols].std()].sum()
                outliers.value = f"OUTLIERS:\n{outliers_data.to_string()}"
                
                page.update()

        file_picker = ft.FilePicker(on_result=on_file_upload)
        page.overlay.append(file_picker)
        
        data_head = ft.Text("")
        data_tail = ft.Text("")
        data_describe = ft.Text("")
        missing_values = ft.Text("")
        outliers = ft.Text("")
        
        return ft.Column([
            ft.Text("Upload your dataset:"),
            ft.ElevatedButton("Browse", on_click=lambda _: file_picker.pick_files(allow_multiple=False)),
            ft.Divider(),
            data_head,
            ft.Divider(),
            data_tail,
            ft.Divider(),
            data_describe,
            ft.Divider(),
            missing_values,
            ft.Divider(),
            outliers,
        ], spacing=5, scroll="auto")

    # Define the model selection interface
    def model_selection_interface():
        selected_model_text = ft.Text("")

        def select_model(model_type):
            global selected_model, model
            selected_model = model_type
            selected_model_text.value = f"Selected classification Model: {model_type}"
            model = build_model(data, selected_model)
            page.update()

        return ft.Column([
            ft.Text("Select a model:"),
            ft.Row([
                ft.ElevatedButton("Random Forest", on_click=lambda _: select_model("Random Forest")),
                ft.ElevatedButton("Decision Tree", on_click=lambda _: select_model("Decision Tree")),
                ft.ElevatedButton("Gradient boost", on_click=lambda _: select_model("Gradient boost")),
                
                # Add more buttons for other models as needed
            ]),
            ft.Divider(),
            selected_model_text,
        ], spacing=5, scroll="auto")

    # Define the classification interface
    def classification_interface():
        accuracy_output = ft.Text("")
        predicted_results = ft.Text("")

        def classify(_):
            global data, model
            if data is not None and model is not None:
                x = data.drop('Loan_Status', axis=1)
                y = data['Loan_Status']
                x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)
                
                y_pred = model.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred)
                report=classification_report(y_test,y_pred)
                
                accuracy_output.value = f"Classifier: {selected_model}\nAccuracy: {accuracy} \n Report: {report}"
                
                # Display predicted results
                predicted_results_df = pd.concat([x_test.reset_index(drop=True), pd.Series(y_pred, name='Pred_Loan_Status')], axis=1)
                predicted_results.value = f"Predicted Results:\n{predicted_results_df.to_string()}"
                
                page.update()

        return ft.Column([
            ft.ElevatedButton("Classify", on_click=classify),
            ft.Divider(),
            ft.Text("Model accuracy:"),
            accuracy_output,
            ft.Divider(),
            predicted_results,
        ], spacing=10, scroll="auto")

    # Creating tabs
    tabs = ft.Tabs(
        selected_index=0,
        tabs=[
            ft.Tab(
                text="Upload",
                content=upload_interface()
            ),
            ft.Tab(
                text="Select Model",
                content=model_selection_interface()
            ),
            ft.Tab(
                text="Classification",
                content=classification_interface()
            ),
        ],
    )

    page.add(tabs)
    

# Start the Flet app
ft.app(target=main)
