#!/usr/bin/env python3

import argparse
import os
import sys
from pandas import DataFrame
import pandas as pd
from gradient_descent import my_gradient_descent
from describe import describe_dataset
from scatter_plot import my_scatter_plot
from histogram import historigram
from pair_plot import my_pair_plot

def main():
	parser = argparse.ArgumentParser(description="Programa principal de dslr")
	parser.add_argument(
		"-v", "--verbose",
		action="store_true",
		help="activar salida detallada"
	)
	# MARIO INI - Cambio de --csv a argumento posicional para consistencia
	parser.add_argument(
		"dataset",
		type=str,
		nargs='?',
		default=None,
		help="Ruta al archivo CSV"
	)
	parser.add_argument(
		"--csv",
		type=str,
		required=False,
		help="Ruta al archivo CSV (deprecated, usar argumento posicional)"
	)
	# MARIO FIN
	args = parser.parse_args()

	if args.verbose:
		print("Modo verbose activado")
	
	# MARIO INI - Soportar ambos formatos de argumento
	csv_path = args.dataset if args.dataset else args.csv
	if not csv_path:
		print("Error: Debe proporcionar la ruta al archivo CSV")
		return
	
	# MARIO INI - Validaciones
	# Verificar si el archivo existe
	if not os.path.exists(csv_path):
		print(f"Error: El archivo '{csv_path}' no existe")
		sys.exit(1)
	
	# Verificar si es un archivo (no directorio)
	if not os.path.isfile(csv_path):
		print(f"Error: '{csv_path}' no es un archivo")
		sys.exit(1)
	
	# Verificar si el archivo está vacío
	if os.path.getsize(csv_path) == 0:
		print(f"Error: El archivo '{csv_path}' está vacío")
		sys.exit(1)
	
	# Verificar extensión .csv
	if not csv_path.lower().endswith('.csv'):
		print(f"Error: El archivo debe tener extensión .csv")
		sys.exit(1)
	# MARIO FIN
	
	try:
		dataset_pandas: DataFrame = pd.read_csv(csv_path)
	except pd.errors.EmptyDataError:
		print(f"Error: El archivo CSV está vacío o no tiene datos válidos")
		sys.exit(1)
	except pd.errors.ParserError:
		print(f"Error: El archivo CSV tiene un formato inválido")
		sys.exit(1)
	except Exception as e:
		print(f"Error al leer el archivo CSV: {e}")
		sys.exit(1)
	
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
	# MARIO FIN
	df_describe = describe_dataset(dataset_pandas)
	df_ravenclaw = dataset_pandas[dataset_pandas['Hogwarts House'] == 'Ravenclaw']
	df_slytherin = dataset_pandas[dataset_pandas['Hogwarts House'] == 'Slytherin']
	df_gryffindor = dataset_pandas[dataset_pandas['Hogwarts House'] == 'Gryffindor']
	df_hufflepuff = dataset_pandas[dataset_pandas['Hogwarts House'] == 'Hufflepuff']
	datasets = [df_ravenclaw, df_slytherin, df_gryffindor, df_hufflepuff]
	historigram(datasets)
	my_scatter_plot(datasets)
	# MARIO INI - pair_plot devuelve datos filtrados, pero gradient_descent necesita datos originales
	# my_pair_plot(dataset_pandas)
	# data = my_pair_plot(dataset_pandas)
	my_pair_plot(dataset_pandas)
	data = dataset_pandas.select_dtypes(include='number').copy()
	if "Index" in data.columns:
		data = data.drop(columns=["Index"])
	# MARIO FIN
	my_gradient_descent(data, df_describe, dataset_pandas)

if __name__ == "__main__":
	main()