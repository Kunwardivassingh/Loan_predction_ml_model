import flet as ft
import pandas as pd
from io import StringIO

def main(page: ft.Page):
    page.title = "CSV File Uploader"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    # uploaded file content
    uploaded_file_content = None

    #  file picker
    file_picker = ft.FilePicker(on_result=lambda e: file_picker_result(e))
    page.overlay.append(file_picker)

    # display the uploaded file name
    file_name_text = ft.Text()

    #  display the data
    data_display = ft.Container(
        content=ft.Column([], scroll="always"),
        width="100%",
        height=400,
        border=ft.border.all(1, ft.colors.BLACK),
        padding=10,
        bgcolor=ft.colors.WHITE,
    )

    # file picker result
    def file_picker_result(e):
        nonlocal uploaded_file_content
        if e.files:
            file = e.files[0]
            file_name_text.value = f"File selected: {file.name}"
            with open(file.path, 'r') as f:
                uploaded_file_content = f.read()
            page.update()

    # Save the uploaded file 
    def upload_file(e):
        if uploaded_file_content:
            global uploaded_file_df
            uploaded_file_df = pd.read_csv(StringIO(uploaded_file_content))
            print("File uploaded successfully!")
            file_name_text.value = f"File uploaded: {file_name_text.value.split(': ')[1]}"
            show_data(None)  # Automatically show data after upload
            page.update()

    # Display file content
    def show_data(e):
        if uploaded_file_content:
            df = pd.read_csv(StringIO(uploaded_file_content))
            data_display.content.controls.clear()

            headers = [ft.DataColumn(ft.Text(col, size=8, weight=ft.FontWeight.BOLD, color=ft.colors.BLACK)) for col in df.columns]
            rows = []

            for i in range(len(df)):
                row = []
                for col in df.columns:
                    cell_value = df.iloc[i][col]
                    row.append(ft.DataCell(ft.Text(str(cell_value), size=8, color=ft.colors.BLACK)))
                rows.append(ft.DataRow(cells=row))

            data_table = ft.DataTable(columns=headers, rows=rows)
            data_display.content.controls.append(data_table)
            page.update()

    select_button = ft.ElevatedButton("Select CSV File", on_click=lambda e: file_picker.pick_files())

    upload_button = ft.ElevatedButton("Upload CSV File", on_click=upload_file)

    show_button = ft.ElevatedButton("Show Data", on_click=show_data)

    page.add(
        ft.Column(
            [
                ft.Text("Select a CSV file:"),
                ft.Row([select_button, file_name_text]),
                ft.Row([upload_button, show_button]),
                data_display,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        )
    )

ft.app(target=main)
