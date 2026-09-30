import seaborn as sns
from pandas import DataFrame
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys
import os

def my_pair_plot(datasets: DataFrame, save_path: str = None, sample_n: int = 500, show: bool = True) -> DataFrame:
	# MARIO INI - Validaciones
	if datasets is None or datasets.empty:
		print(f"Error: El DataFrame está vacío o es None")
		return datasets
	
	if len(datasets) == 0:
		print(f"Error: El DataFrame no tiene filas")
		return datasets
	
	if "Hogwarts House" not in datasets.columns:
		print(f"Error: El DataFrame debe contener la columna 'Hogwarts House'")
		return datasets
	
	# Verificar si hay columnas numéricas
	numeric_cols = datasets.select_dtypes(include='number').columns.tolist()
	if len(numeric_cols) == 0:
		print(f"Error: No se encontraron columnas numéricas en el DataFrame")
		return datasets
	# MARIO FIN
	
	colors = ['blue', 'green', 'red', 'orange']
	data = datasets
	if sample_n and len(datasets) > sample_n:
		data = datasets.sample(sample_n, random_state=0)

	threshold = 0.8
	columnas_numericas = data.select_dtypes(include='number').columns.tolist()
	columnas_a_eliminar = set()
	for col1 in columnas_numericas:
		if col1 in columnas_a_eliminar or col1 == "Index":
			continue
		for col2 in columnas_numericas:
			if col1 == col2 or col2 == "Index" or col2 in columnas_a_eliminar:
				continue
			c1 = col1
			c2 = col2
			clean = data[[c1, c2]].dropna()
			if len(clean) > 2:
				corr, pval = pearsonr(clean[c1], clean[c2])
				if abs(corr) >= threshold:
					columnas_a_eliminar.add(col2)

	data = data.drop(columns=list(columnas_a_eliminar))

	remaining_numeric = [col for col in data.select_dtypes(include='number').columns.tolist() if col != "Index"]
	if len(remaining_numeric) == 0:
		print(f"Error: No hay columnas numéricas suficientes después de eliminar correlaciones")
		return data

	if "Index" in data.columns:
		data = data.drop(columns="Index")
	g = sns.pairplot(data, hue="Hogwarts House", palette=colors)

	if save_path:
		try:
			g.savefig(save_path)
		except Exception:
			pass

	if show:
		plt.show()
	else:
		plt.close()
	return data

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Pair plot of Hogwarts features")
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
		
		my_pair_plot(dataset)
	except pd.errors.EmptyDataError:
		print(f"Error: El archivo CSV está vacío o no tiene datos válidos")
		sys.exit(1)
	except pd.errors.ParserError:
		print(f"Error: El archivo CSV tiene un formato inválido")
		sys.exit(1)
	except Exception as e:
		print(f"Error: {e}")
		sys.exit(1)