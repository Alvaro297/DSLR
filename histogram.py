import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.stats import kruskal

def is_homogeneus(df_ravenclaw:DataFrame, df_slytherin: DataFrame, df_gryffindor: DataFrame, df_hufflepuff: DataFrame, col:str) -> bool:
	# MARIO INI - Validaciones
	if df_ravenclaw.empty or df_slytherin.empty or df_gryffindor.empty or df_hufflepuff.empty:
		print(f"Advertencia: Uno o más DataFrames están vacíos, omitiendo columna {col}")
		return False
	
	if col not in df_ravenclaw.columns:
		print(f"Advertencia: La columna '{col}' no existe en los DataFrames")
		return False
	
	# Verificar si hay suficientes datos después de dropna
	ravenclaw_data = df_ravenclaw[col].dropna()
	slytherin_data = df_slytherin[col].dropna()
	gryffindor_data = df_gryffindor[col].dropna()
	hufflepuff_data = df_hufflepuff[col].dropna()
	
	MIN_SAMPLES = 2
	if len(ravenclaw_data) < MIN_SAMPLES or len(slytherin_data) < MIN_SAMPLES or \
	   len(gryffindor_data) < MIN_SAMPLES or len(hufflepuff_data) < MIN_SAMPLES:
		print(f"Advertencia: No hay suficientes datos en la columna {col} para el test de Kruskal")
		return False
	# MARIO FIN
	
	stat, p_valor = kruskal(ravenclaw_data, slytherin_data, gryffindor_data, hufflepuff_data)
	if p_valor > 0.05:
		print(f"Las distribuciones son homogéneas en la columna {col} (p-valor: {p_valor:.4f})")
		return True
	else:
		print(f"Las distribuciones son diferentes (p-valor: {p_valor:.4f})")
		return False

def historigram(datasets: list):
	# MARIO INI - Validaciones
	if len(datasets) != 4:
		print(f"Error: Se esperan 4 DataFrames, pero se recibieron {len(datasets)}")
		return
	
	df_ravenclaw, df_slytherin, df_gryffindor, df_hufflepuff = datasets
	
	# Verificar si hay columnas numéricas
	columnas_numericas = df_ravenclaw.select_dtypes(include='number').columns.tolist()
	if len(columnas_numericas) == 0:
		print(f"Error: No se encontraron columnas numéricas en los DataFrames")
		return
	# MARIO FIN
	
	for col in columnas_numericas:
		if col == "Index":
			continue
		if is_homogeneus(df_ravenclaw, df_slytherin, df_gryffindor, df_hufflepuff, col):
			plt.hist([df_ravenclaw[col], df_slytherin[col], df_gryffindor[col], df_hufflepuff[col]],
					bins=20, alpha=0.5, label=['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff'])
			plt.legend()
			plt.show()