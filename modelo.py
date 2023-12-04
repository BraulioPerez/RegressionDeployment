import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class ProcesoDatos():

    def __init__(self, datos_limpieza, datos_entrenamiento, test_size):
        # Inicializar las instancias de LimpiezaDatos y Entrenamiento
        self.limpieza = LimpiezaDatos(datos_limpieza)
        self.entrenamiento = Entrenamiento(datos_entrenamiento, test_size)

    def ejecutar_proceso(self):
        # Realizar la limpieza, normalización y entrenamiento de datos
        datos_limpios = self.limpieza.nulos_normalizacion()
        datos_sin_outliers = datos_limpios.outliers_delete()
        modelo_entrenado = self.entrenamiento.entrenar(datos_sin_outliers)
        return modelo_entrenado

class LimpiezaDatos():

    def __init__(self, datos, q1, q3):
        # Inicializar con el archivo de datos, q1 y q3 para la limpieza
        self.q1 = q1
        self.q3 = q3
        self.datos = pd.read_csv(datos)

    def Super_limpieza(self):
        # Eliminación de nulos
        self.datos.dropna(inplace=True)

        # Transformación de la columna median_income a float
        self.datos["median_income"] = self.datos["median_income"].apply(lambda x: x.replace(" ", ""))
        self.datos["median_income"] = self.datos["median_income"].astype("float")

        # Aplicar logaritmo a ciertas columnas para obtener una distribución normal
        self.datos["total_rooms"] = np.log(self.datos["total_rooms"] + 1)
        self.datos["total_bedrooms"] = np.log(self.datos["total_bedrooms"] + 1)
        self.datos["population"] = np.log(self.datos["population"] + 1)
        self.datos["households"] = np.log(self.datos["households"] + 1)

        # Codificación ordinal en la columna ocean proximity y eliminación de ISLAND
        self.datos = self.datos.join(pd.get_dummies(self.datos.ocean_proximity, dtype=int)).drop(["ocean_proximity"], axis=1)
        self.datos = self.datos[self.datos["ISLAND"] != 1]
        self.datos = self.datos.drop(["ISLAND"], axis=1)
        self.datos["bedroom_ratio"] = self.datos["total_bedrooms"] / self.datos["total_rooms"]
        self.datos["household_rooms"] = self.datos["total_rooms"] / self.datos["households"]

        # Eliminación de outliers
        q1 = self.datos.quantile(self.q1)
        q3 = self.datos.quantile(self.q3)
        IQR = q3 - q1 
        lower_bound = q1 - 1.5 * IQR
        upper_bound = q3 + 1.5 * IQR
        outliers = (self.datos < lower_bound) | (self.datos > upper_bound)
        outliers_count = outliers.sum(axis=1)
        self.datos = self.datos[outliers_count == 0]

        # Guardar el DataFrame limpio en un archivo CSV
        self.datos.to_csv("dataset_muy_limpio.csv", index=False)

        # Devolver el DataFrame limpio
        return self.datos

class Entrenamiento():

    def __init__(self, datos, test_size):
        # Inicializar con el archivo de datos y el tamaño del conjunto de prueba
        self.datos = pd.read_csv(datos)
        self.test_size = test_size

    def entrenar(self):
        # Excluir el 5% de los datos para evaluar el rendimiento
        five_percent = self.datos.sample(frac=0.05, random_state=42)
        self.datos = self.datos.drop(five_percent.index)

        # Seleccionar columnas importantes para la predicción
        features = self.datos.drop(["median_house_value"], axis=1)
        target = self.datos["median_house_value"]

        # Dividir el conjunto de datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=self.test_size, random_state=42)

        # Normalizar las características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Crear y entrenar el modelo de regresión lineal
        model = LinearRegression()
        model = model.fit(X_train_scaled, y_train)

        return model

class Prediccion():

    def __init__(self, lon, lat, h_m_a, t_rooms, t_bed, pop, house, m_i, modelo, ocean_distance):
        # Crear un array con datos de usuario para predecir
        bed_ra = t_bed / t_rooms
        h_rooms = t_rooms / house
        data = [lon, lat, h_m_a, np.log(t_rooms + 1), np.log(t_bed + 1), np.log(pop + 1), np.log(house + 1), m_i]

        if ocean_distance == 1:
            data.extend([0, 1, 0, 0, bed_ra, h_rooms])
        elif ocean_distance == 2:
            data.extend([1, 0, 0, 0, bed_ra, h_rooms])
        elif ocean_distance == 3:
            data.extend([0, 0, 1, 0, bed_ra, h_rooms])
        else:
            data.extend([0, 0, 0, 1, bed_ra, h_rooms])

        self.user_data = np.array(data).reshape(1, -1)
        self.modelo = modelo
        
    def Predecir(self):
        # Realizar la predicción con el modelo entrenado
        resultado = self.modelo.predict(self.user_data)
        return resultado[0]
