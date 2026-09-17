import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.stats import pearsonr
import pandas as pd

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
			# MARIO INI - Validaciones
			MIN_SAMPLES = 3
			if len(clean) < MIN_SAMPLES:
				continue
			# MARIO FIN
			if len(clean) > 2:
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