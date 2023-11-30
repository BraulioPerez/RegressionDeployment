from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import os
from modelo import LimpiezaDatos, Entrenamiento, Prediccion


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Process the uploaded CSV file (e.g., read into a DataFrame)
        # You can perform additional processing or analysis on 'df' if needed.

        return render_template('index.html', message='File uploaded successfully')

    return render_template('index.html', message='Invalid file type')


@app.route('/clean', methods=['POST'])
def clean_file():
    # Load the uploaded CSV file
    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'housing.csv')
    
    q1_value = float(request.form["q1_value"]) * 0.01
    q3_value = float(request.form["q3_value"]) * 0.01

    df = LimpiezaDatos(datos=filename, q1=0.2, q3=0.8)
    no_null_df = df.nulos_normalizacion()
    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_limpio.csv')
    no_null_df = LimpiezaDatos(datos=filename, q1=q1_value, q3 = q3_value)
    cleaned_df = no_null_df.outliers_delete()

    # Save the cleaned DataFrame back to the CSV file
    cleaned_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_muy_limpio.csv')
    cleaned_df.to_csv(cleaned_filename, index=False)

    return redirect(url_for('index'))


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
