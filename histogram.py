import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.stats import kruskal
import pandas as pd
import argparse
import sys
import os

def is_homogeneus(df_ravenclaw:DataFrame, df_slytherin: DataFrame, df_gryffindor: DataFrame, df_hufflepuff: DataFrame, col:str) -> bool:
	if df_ravenclaw.empty or df_slytherin.empty or df_gryffindor.empty or df_hufflepuff.empty:
		return False
	
	if col not in df_ravenclaw.columns:
		return False
	
	ravenclaw_data = df_ravenclaw[col].dropna()
	slytherin_data = df_slytherin[col].dropna()
	gryffindor_data = df_gryffindor[col].dropna()
	hufflepuff_data = df_hufflepuff[col].dropna()
	
	MIN_SAMPLES = 2
	if len(ravenclaw_data) < MIN_SAMPLES or len(slytherin_data) < MIN_SAMPLES or \
	   len(gryffindor_data) < MIN_SAMPLES or len(hufflepuff_data) < MIN_SAMPLES:
		return False
	
	stat, p_valor = kruskal(ravenclaw_data, slytherin_data, gryffindor_data, hufflepuff_data)
	return p_valor > 0.05

def historigram(datasets: list):
	if len(datasets) != 4:
		print(f"Error: Se esperan 4 DataFrames, pero se recibieron {len(datasets)}")
		return
	
	df_ravenclaw, df_slytherin, df_gryffindor, df_hufflepuff = datasets
	
	columnas_numericas = df_ravenclaw.select_dtypes(include='number').columns.tolist()
	if len(columnas_numericas) == 0:
		print(f"Error: No se encontraron columnas numéricas en los DataFrames")
		return
	
	homogeneous_cols = []
	for col in columnas_numericas:
		if col == "Index":
			continue
		if is_homogeneus(df_ravenclaw, df_slytherin, df_gryffindor, df_hufflepuff, col):
			homogeneous_cols.append(col)
	
	if not homogeneous_cols:
		print("No se encontraron distribuciones homogéneas entre las casas.")
		return
	
	print(f"\nColumnas con distribución homogénea entre casas: {', '.join(homogeneous_cols)}")
	
	for col in homogeneous_cols:
		plt.hist([df_ravenclaw[col], df_slytherin[col], df_gryffindor[col], df_hufflepuff[col]],
				bins=20, alpha=0.5, label=['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff'])
		plt.legend()
		plt.title(f"Distribución homogénea: {col}")
		plt.xlabel(col)
		plt.ylabel("Frecuencia")
		plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Histogram of Hogwarts courses")
	parser.add_argument("dataset", type=str, help="Ruta al archivo CSV")
	args = parser.parse_args()
	
	dataset_path = args.dataset
	
	if not os.path.exists(dataset_path):
		print(f"Error: El archivo '{dataset_path}' no existe")
		sys.exit(1)
	
	if not os.path.isfile(dataset_path):
		print(f"Error: '{dataset_path}' no es un archivo")
		sys.exit(1)
	
	if os.path.getsize(dataset_path) == 0:
		print(f"Error: El archivo '{dataset_path}' está vacío")
		sys.exit(1)
	
	if not dataset_path.lower().endswith('.csv'):
		print(f"Error: El archivo debe tener extensión .csv")
		sys.exit(1)
	
	try:
		dataset = pd.read_csv(dataset_path)
		
		if dataset.empty or len(dataset) == 0:
			print(f"Error: El dataset está vacío")
			sys.exit(1)
		
		if "Hogwarts House" not in dataset.columns:
			print(f"Error: El dataset debe contener la columna 'Hogwarts House'")
			sys.exit(1)
		
		df_ravenclaw = dataset[dataset['Hogwarts House'] == 'Ravenclaw']
		df_slytherin = dataset[dataset['Hogwarts House'] == 'Slytherin']
		df_gryffindor = dataset[dataset['Hogwarts House'] == 'Gryffindor']
		df_hufflepuff = dataset[dataset['Hogwarts House'] == 'Hufflepuff']
		datasets = [df_ravenclaw, df_slytherin, df_gryffindor, df_hufflepuff]
		
		historigram(datasets)
	except pd.errors.EmptyDataError:
		print(f"Error: El archivo CSV está vacío o no tiene datos válidos")
		sys.exit(1)
	except pd.errors.ParserError:
		print(f"Error: El archivo CSV tiene un formato inválido")
		sys.exit(1)
	except Exception as e:
		print(f"Error: {e}")
		sys.exit(1)