import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class ProcesoDatos():

    def __init__(self, datos_limpieza, datos_entrenamiento, test_size):
        self.limpieza = LimpiezaDatos(datos_limpieza)
        self.entrenamiento = Entrenamiento(datos_entrenamiento, test_size)

    def ejecutar_proceso(self):
        datos_limpios = self.limpieza.nulos_normalizacion()
        datos_sin_outliers = datos_limpios.outliers_delete()
        modelo_entrenado = self.entrenamiento.entrenar(datos_sin_outliers)
        return modelo_entrenado

class LimpiezaDatos():

    def __init__(self, datos, q1, q3):
        self.q1 = q1 or 0.2
        self.q3 = q3 or 0.8
        self.datos = pd.read_csv(datos)

    def nulos_normalizacion(self):
        self.datos.dropna(inplace=True)
        self.datos["total_rooms"] = np.log(self.datos["total_rooms"] + 1)
        self.datos["total_bedrooms"] = np.log(self.datos["total_bedrooms"] + 1)
        self.datos["population"] = np.log(self.datos["population"] + 1)
        self.datos["households"] = np.log(self.datos["households"] + 1)
        self.datos = self.datos.join(pd.get_dummies(self.datos.ocean_proximity, dtype=int)).drop(["ocean_proximity"], axis=1)
        self.datos["bedroom_ratio"] = self.datos["total_bedrooms"] / self.datos["total_rooms"]
        self.datos["household_rooms"] = self.datos["total_rooms"] / self.datos["households"]
        self.datos.to_csv("dataset_limpio.csv")
        return self.datos

    def outliers_delete(self):
        self.datos = pd.read_csv("dataset_limpio.csv")
        # Eliminacion de los datos de Island
        self.datos = self.datos.drop(["ISLAND"], axis=1)  
        # Assign the result back to the variable

        # Eliminacion de outliers y el calculo el rango intercuartilico (IQR)
        
        IQR = self.q3 - self.q1

        # Definir los limites inferior y superior para identificar outliers
        lower_bound = self.q1 - 1.5 * IQR
        upper_bound = self.q3 + 1.5 * IQR
        # Filtrar filas que estan dentro de los limites
        outliers = (self.datos < lower_bound) | (self.datos > upper_bound)

        # Contar el numero de outliers en cada fila
        outliers_count = outliers.sum(axis=1)

        # Eliminar filas que contienen al menos un outlier
        self.datos = self.datos[outliers_count == 0]
        self.datos.to_csv("dataset_muy_limpio.csv")
        return self.datos

class Entrenamiento():

    def __init__(self, datos, test_size):
        self.datos = pd.read_csv(datos)
        self.test_size = test_size

    def entrenar(self):
        # Seleccionar el 5% para excluirlo
        five_percent = self.datos.sample(frac=0.05, random_state=42)
        self.datos = self.datos.drop(five_percent.index)

        # Seleccionar las columnas importantes para la prediccion
        # Ajustar la lista segun los criterios
        # Seleccionamos todas las columnas excepto "median_house_value"
        features = self.datos.drop(["median_house_value"], axis=1)
        # La columna 'median_house_value' es nuestra variable de objetivo
        target = self.datos["median_house_value"]

        # Iniciamos la division para hacer nuestro entrenamiento
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=self.test_size, random_state=42)

        # Normalizamos las caracteristicas para asegurar que todos tengan la misma escala
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Creamos el modelo de regresio lineal
        model = LinearRegression()

        # Entrenamos al modelo con los datos de entrenamiento
        model.fit(X_train_scaled, y_train)
        return model




class Prediccion():

    def __init__(self, lon, lat, h_m_a, t_rooms, t_bed, h_rooms, pop, house, m_i, bed_in, modelo):
        self.user_data = np.array([lon,lat,h_m_a,t_rooms,t_bed,h_rooms,pop, house, m_i, bed_in]).reshape(1, -1)
        self.modelo = modelo

    def Predecir(self):
        resultado = self.modelo.predict(self.user_data)
        return resultado[0]
    