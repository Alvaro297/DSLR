#!/usr/bin/env python3

# MARIO INI - Archivo creado para cumplir con el subject
import argparse
import sys
import pandas as pd
import os
from pandas import DataFrame
from describe import describe_dataset
from gradient_descent import my_gradient_descent

def main():
	parser = argparse.ArgumentParser(description="Entrena el modelo de regresion logistica")
	parser.add_argument("dataset", type=str, help="Ruta al archivo CSV de entrenamiento")
	args = parser.parse_args()
	
	# MARIO INI - Validaciones
	dataset_path = args.dataset
	
	# Verificar si el archivo existe
	if not os.path.exists(dataset_path):
		print(f"Error: El archivo '{dataset_path}' no existe")
		sys.exit(1)
	
	# Verificar si es un archivo (no directorio)
	if not os.path.isfile(dataset_path):
		print(f"Error: '{dataset_path}' no es un archivo")
		sys.exit(1)
	
	# Verificar si el archivo está vacío
	if os.path.getsize(dataset_path) == 0:
		print(f"Error: El archivo '{dataset_path}' está vacío")
		sys.exit(1)
	
	# Verificar extensión .csv
	if not dataset_path.lower().endswith('.csv'):
		print(f"Error: El archivo debe tener extensión .csv")
		sys.exit(1)
	# MARIO FIN
	
	try:
		# Cargar dataset
		dataset_pandas: DataFrame = pd.read_csv(dataset_path)
		
		# MARIO INI - Validaciones del DataFrame
		if dataset_pandas.empty:
			print(f"Error: El dataset está vacío")
			sys.exit(1)
		
		if len(dataset_pandas) == 0:
			print(f"Error: El dataset no tiene filas")
			sys.exit(1)
		
		# Verificar columna "Hogwarts House"
		if "Hogwarts House" not in dataset_pandas.columns:
			print(f"Error: El dataset debe contener la columna 'Hogwarts House'")
			sys.exit(1)
		
		# Verificar valores válidos en "Hogwarts House"
		valid_houses = {'Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff'}
		unique_houses = set(dataset_pandas["Hogwarts House"].dropna().unique())
		if not unique_houses.issubset(valid_houses):
			invalid = unique_houses - valid_houses
			print(f"Error: Se encontraron valores inválidos en 'Hogwarts House': {invalid}")
			sys.exit(1)
		
		# Verificar si hay columnas numéricas
		numeric_cols = dataset_pandas.select_dtypes(include='number').columns.tolist()
		if len(numeric_cols) == 0:
			print(f"Error: No se encontraron columnas numéricas en el dataset")
			sys.exit(1)
		
		# Verificar si hay suficientes filas para entrenar
		MIN_ROWS = 10
		if len(dataset_pandas) < MIN_ROWS:
			print(f"Error: Se necesitan al menos {MIN_ROWS} filas para entrenar, pero solo hay {len(dataset_pandas)}")
			sys.exit(1)
		
		print(f"Dataset cargado: {len(dataset_pandas)} filas")
		# MARIO FIN
		
		# Calcular estadisticas para normalizacion
		print("\nCalculando estadisticas...")
		df_describe = describe_dataset(dataset_pandas)
		
		# Preparar datos para entrenamiento
		# Eliminar columnas no numericas y la columna objetivo
		data = dataset_pandas.select_dtypes(include='number').copy()
		if "Index" in data.columns:
			data = data.drop(columns=["Index"])
		
		# Entrenar modelo
		print("\nEntrenando modelo con gradient descent...")
		my_gradient_descent(data, df_describe, dataset_pandas)
		print("\nEntrenamiento completado. Modelo guardado en model.json")
		
	except pd.errors.EmptyDataError:
		print(f"Error: El archivo CSV está vacío o no tiene datos válidos")
		sys.exit(1)
	except pd.errors.ParserError:
		print(f"Error: El archivo CSV tiene un formato inválido")
		sys.exit(1)
	except FileNotFoundError:
		print(f"Error: No se encontro el archivo {dataset_path}")
		sys.exit(1)
	except Exception as e:
		print(f"Error: {e}")
		sys.exit(1)

if __name__ == "__main__":
	main()
# MARIO FIN
