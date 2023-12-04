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

@app.route('/upload', methods=["POST", "GET"])
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


@app.route('/clean', methods=["POST", "GET"])
def clean_file():
    # Cargamos la dirección del archivo
    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'housing.csv')
    
    # cargamos los valores de q1 y q3 para la limpieza
    q1_value = float(request.form["q1_value"]) * 0.01
    q3_value = float(request.form["q3_value"]) * 0.01

    # Asignamos el housing.csv a nuestro dataframe como df y nuestra clase
    df = LimpiezaDatos(datos=filename, q1=float(q1_value), q3=float(q3_value))
    # Aplicamos la eliminación de nulos
    df = df.Super_limpieza()
    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_limpio.csv')

    return redirect(url_for('index'))

model = None

@app.route('/train', methods=["POST", "GET"])
def train_model():
    # Seleccionamos el archivo de datos limpio
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, "dataset_muy_limpio.csv")

    # Extraemos tamaño del test desde html
    test_size = float(request.form["test_size"]) * 0.01

    # Se asigna el modelo a una variable y se le pasa el método de entrenar
    modelo = Entrenamiento(datos = file_path, test_size= test_size)
    modelo_final = modelo.entrenar()
    global model
    model = modelo_final

    return redirect(url_for('index'))


prediccion_lista = None
model = model

@app.route('/predict', methods=["POST", "GET"])
def predict():
    global model
    global prediccion_lista
    ocean_distance = float(request.form["ocean_distance"])
    latitude = float(request.form["latitude"])
    longitude = float(request.form["longitude"])
    housing_median_age = float(request.form["housing_median_age"])
    total_bedrooms = float(request.form["total_bedrooms"])
    total_rooms = float(request.form["total_rooms"])
    population = float(request.form["population"])
    households = float(request.form["households"])
    median_income = float(request.form["median_income"])

    #bedroom_ratio,household_rooms

    resultado = Prediccion(modelo=model, lon=longitude,lat=latitude,h_m_a=housing_median_age,t_rooms=total_rooms, t_bed=total_bedrooms, pop=population,house=households, m_i=median_income, ocean_distance=ocean_distance)    
    valor_predicho =  resultado.Predecir()

    return render_template("index.html", prediction_message=f"La casa vale ${valor_predicho} pesos")





if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000, host="0.0.0.0")