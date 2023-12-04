import pandas as pd 

df = pd.read_csv("housing.csv")
df = df[["total_rooms","total_bedrooms","population","households"]]

q1 = df.quantile(0.3)
q3 = df.quantile(0.7)

IQR = q3 - q1 

        # Definir los límites inferior y superior para identificar outliers
lower_bound = q1 - 1.5 * IQR
upper_bound = q3 + 1.5 * IQR

        # Filtrar filas que están dentro de los límites
outliers = (df < lower_bound) | (df > upper_bound)

        # Contar el número de outliers en cada fila
outliers_count = outliers.sum(axis=1)

        # Eliminar filas que contienen al menos un outlier
df = df[outliers_count == 0]


rangos = df.agg(["min", "max"])
print(df)
print(rangos)