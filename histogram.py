import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.stats import kruskal

def is_homogeneus(df_ravenclaw:DataFrame, df_slytherin: DataFrame, df_gryffindor: DataFrame, df_hufflepuff: DataFrame, col:str) -> bool:
	stat, p_valor = kruskal(df_ravenclaw[col].dropna(), df_slytherin[col].dropna(), df_gryffindor[col].dropna(), df_hufflepuff[col].dropna())
	if p_valor > 0.05:
		print(f"Las distribuciones son homogéneas en la columna {col} (p-valor: {p_valor:.4f})")
		return True
	else:
		print(f"Las distribuciones son diferentes (p-valor: {p_valor:.4f})")
		return False

def historigram(datasets: list):
	df_ravenclaw, df_slytherin, df_gryffindor, df_hufflepuff = datasets
	columnas_numericas = df_ravenclaw.select_dtypes(include='number').columns.tolist()
	for col in columnas_numericas:
		if col == "Index":
			continue
		if is_homogeneus(df_ravenclaw, df_slytherin, df_gryffindor, df_hufflepuff, col):
			plt.hist([df_ravenclaw[col], df_slytherin[col], df_gryffindor[col], df_hufflepuff[col]],
					bins=20, alpha=0.5, label=['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff'])
			plt.legend()
			plt.show()