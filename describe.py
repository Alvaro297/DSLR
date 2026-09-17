#!/usr/bin/env python3

import pandas as pd
from pandas import DataFrame
import math
import argparse  # MARIO INI
import sys
import os  # MARIO FIN

def my_percent(dataset: DataFrame, column: str, percent: int):
	values = []
	for index, row in dataset.iterrows():
		val = row[column]
		if pd.notna(val):
			values.append(val)
	if len(values) == 0:
		return float('nan')
	values.sort()
	n = len(values)
	position = (percent / 100.0) * (n - 1)
	if position == int(position):
		return float(values[int(position)])
	lower_index = int(position)
	upper_index = lower_index + 1
	fraction = position - lower_index
	lower_value = values[lower_index]
	upper_value = values[upper_index]
	return float(lower_value + fraction * (upper_value - lower_value))

def my_max(dataset: DataFrame, column: str):
	max = 0
	flag: bool = False
	for index, row in dataset.iterrows():
		val = row[column]
		if pd.notna(val) and flag is False:
			max = val
			flag = True
		elif pd.notna(val) and max < val:
			max = val
	if flag is False:
		return float('nan')
	return max

def my_min(dataset: DataFrame, column: str):
	min = 0
	flag: bool = False
	for index, row in dataset.iterrows():
		val = row[column]
		if pd.notna(val) and flag is False:
			min = val
			flag = True
		elif pd.notna(val) and min > val:
			min = val
	if flag is False:
		return float('nan')
	return min

def my_std(dataset: DataFrame, column: str):
	mean = my_mean(dataset, column)
	if math.isnan(mean):
		return float('nan')
	std_sum = 0
	count: int = 0
	for index, row in dataset.iterrows():
		val = row[column]
		if pd.notna(val):
			std_sum += (val - mean)**2
			count += 1
	std = math.sqrt(std_sum / (count - 1))
	return std

def my_count(dataset: DataFrame, column: str):
	count: int = 0
	for index, row in dataset.iterrows():
		val = row[column]
		if pd.notna(val):
			count += 1
	if count == 0:
		return float(0.0)
	return count

def my_mean(dataset: DataFrame, column: str):
	count: int = 0
	total: int = 0
	for index, row in dataset.iterrows():
		val = row[column]
		if pd.notna(val):
			count += 1
			total += row[column]
	if count == 0:
		return float('nan')
	mean = total / count
	return mean


def describe_dataset(dataset: DataFrame) -> DataFrame:
	result = {}
	columnas_numericas = dataset.select_dtypes(include='number').columns.tolist()
	for col in columnas_numericas:
		result[col] = {
			'Count': my_count(dataset, column=col),
			'Mean': my_mean(dataset, column=col),
			'Std': my_std(dataset, column=col),
			'Min': my_min(dataset, column=col),
			'25%': my_percent(dataset, column=col, percent=25),
			'50%':  my_percent(dataset, column=col, percent=50),
			'75%':  my_percent(dataset, column=col, percent=75),
			'Max': my_max(dataset, column=col)
		}
	df_result = pd.DataFrame(result)
	print(df_result)
	return df_result

# MARIO INI - Validaciones de manejo de errores
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Describe dataset statistics")
	parser.add_argument("dataset", type=str, help="Ruta al archivo CSV")
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
		dataset = pd.read_csv(dataset_path)
		
		# MARIO INI - Validaciones del DataFrame
		if dataset.empty:
			print(f"Error: El dataset está vacío")
			sys.exit(1)
		
		if len(dataset) == 0:
			print(f"Error: El dataset no tiene filas")
			sys.exit(1)
		
		# Verificar si hay columnas numéricas
		numeric_cols = dataset.select_dtypes(include='number').columns.tolist()
		if len(numeric_cols) == 0:
			print(f"Error: No se encontraron columnas numéricas en el dataset")
			sys.exit(1)
		# MARIO FIN
		
		describe_dataset(dataset)
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
# MARIO FIN
