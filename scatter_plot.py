import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.stats import pearsonr
import pandas as pd
import argparse
import sys
import os

def my_scatter_plot(datasets: list):
	# MARIO INI - Validaciones
	if len(datasets) != 4:
		print(f"Error: Se esperan 4 DataFrames, pero se recibieron {len(datasets)}")
		return
	
	df_ravenclaw, df_slytherin, df_gryffindor, df_hufflepuff = datasets
	
	# Verificar si los DataFrames están vacíos
	if df_ravenclaw.empty and df_slytherin.empty and df_gryffindor.empty and df_hufflepuff.empty:
		print(f"Error: Todos los DataFrames están vacíos")
		return
	# MARIO FIN
	
	houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
	colors = ['blue', 'green', 'red', 'orange']

	full = pd.concat([df_ravenclaw, df_slytherin, df_gryffindor, df_hufflepuff], ignore_index=True)
	
	# MARIO INI - Validaciones
	if full.empty:
		print(f"Error: El DataFrame concatenado está vacío")
		return
	# MARIO FIN
	
	numeric_cols = full.select_dtypes(include='number').columns.tolist()
	numeric_cols = [c for c in numeric_cols if c != 'Index']
	if len(numeric_cols) < 2:
		print('No hay suficientes columnas numéricas para calcular correlaciones.')
		return

	threshold = 0.85
	results = []
	for i in range(len(numeric_cols)):
		for j in range(i + 1, len(numeric_cols)):
			c1 = numeric_cols[i]
			c2 = numeric_cols[j]
			clean = full[[c1, c2]].dropna()
			if len(clean) < 3:
				continue
			corr, pval = pearsonr(clean[c1], clean[c2])
			if abs(corr) >= threshold:
				results.append({'Col1': c1, 'Col2': c2, 'r': round(corr, 4), 'p': round(pval, 6)})

	if not results:
		print(f"No se encontraron correlaciones con |r| >= {threshold}")
		return

	df_res = pd.DataFrame(results)
	df_res['abs_r'] = df_res['r'].abs()
	df_res = df_res.sort_values(by='abs_r', ascending=False).drop(columns=['abs_r'])
	print('\n=== Correlaciones de Pearson (dataset completo) ===')
	print(df_res.to_string(index=False))

	for _, row in df_res.iterrows():
		c1 = row['Col1']
		c2 = row['Col2']
		rval = row['r']
		plt.figure(figsize=(10, 6))
		for idx, ds in enumerate([df_ravenclaw, df_slytherin, df_gryffindor, df_hufflepuff]):
			d = ds[[c1, c2]].dropna()
			if d.empty:
				continue
			plt.scatter(d[c1], d[c2], alpha=0.6, label=houses[idx], color=colors[idx], s=40)
		plt.xlabel(c1)
		plt.ylabel(c2)
		plt.title(f"{c1} vs {c2} (r={rval})")
		plt.legend()
		plt.grid(alpha=0.3)
		plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Scatter plot of Hogwarts features")
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
		
		my_scatter_plot(datasets)
	except pd.errors.EmptyDataError:
		print(f"Error: El archivo CSV está vacío o no tiene datos válidos")
		sys.exit(1)
	except pd.errors.ParserError:
		print(f"Error: El archivo CSV tiene un formato inválido")
		sys.exit(1)
	except Exception as e:
		print(f"Error: {e}")
		sys.exit(1)