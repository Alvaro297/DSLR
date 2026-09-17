from pandas import DataFrame
import pandas as pd
from scipy.special import expit
import json
import os
import numpy as np

def my_normalization(data: DataFrame, df_describe: DataFrame):
	data_numeric = data.select_dtypes(include=[np.number])
	fill_dict = {}
	for col in data_numeric.columns:
		try:
			fill_dict[col] = float(df_describe[col]["Mean"])
		except Exception:
			try:
				fill_dict[col] = float(df_describe.at["Mean", col])
			except Exception:
				fill_dict[col] = 0.0
	data_imputed = data_numeric.fillna(fill_dict)
	data_normalized = data_imputed.copy()
	# MARIO INI - Remover prints de debug
	# print(df_describe.columns)
	# print("El data frame")
	# print(data_normalized.columns)
	# MARIO FIN
	for columns in data_normalized.columns:
		# MARIO INI - Remover print de debug
		# print(columns)
		# MARIO FIN
		if columns == "Index":
			continue
		try:
			mean = float(df_describe[columns]["Mean"])
			std = float(df_describe[columns]["Std"])
		except Exception:
			mean = float(df_describe.at["Mean", columns]) if ("Mean" in df_describe.index) else 0.0
			std = float(df_describe.at["Std", columns]) if ("Std" in df_describe.index) else 1.0
		if std == 0.0:
			std = 1.0
		data_normalized[columns] = (data_imputed[columns] - mean) / std
	return data_normalized

def create_labels(df_all: DataFrame):
	houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
	labels = pd.get_dummies(df_all['Hogwarts House'])
	labels = labels.reindex(columns=houses, fill_value=0)
	return labels

def my_gradient_descent(data: DataFrame, df_describe: DataFrame, df_all: DataFrame):
	# MARIO INI - Validaciones
	if data is None or data.empty:
		print(f"Error: El DataFrame de datos está vacío o es None")
		return
	
	if df_describe is None or df_describe.empty:
		print(f"Error: El DataFrame de descripciones está vacío o es None")
		return
	
	if df_all is None or df_all.empty:
		print(f"Error: El DataFrame completo está vacío o es None")
		return
	
	if "Hogwarts House" not in df_all.columns:
		print(f"Error: El DataFrame completo debe contener la columna 'Hogwarts House'")
		return
	
	if len(data) == 0:
		print(f"Error: No hay filas en el DataFrame de datos")
		return
	
	MIN_ROWS = 10
	if len(data) < MIN_ROWS:
		print(f"Error: Se necesitan al menos {MIN_ROWS} filas para entrenar, pero solo hay {len(data)}")
		return
	
	# Verificar si hay columnas numéricas
	if len(data.columns) == 0:
		print(f"Error: No hay columnas en el DataFrame de datos")
		return
	# MARIO FIN
	
	data_normalized = my_normalization(data, df_describe)
	lr = 0.05
	# MARIO INI - Bias por cada clase (one-vs-all)
	# bias = 1.0
	bias = pd.Series(0.0, index=['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff'])
	# MARIO FIN
	labels = create_labels(df_all)
	labels = labels.reindex(index=data_normalized.index, fill_value=0)
	list_weights = pd.DataFrame(
		0,
		index=data_normalized.columns,
		columns=['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
		)
	
	for _ in range(5000):
		z = data_normalized.dot(list_weights) + bias
		pred = expit(z)
		error = pred - labels
		dw = data_normalized.T.dot(error) / len(data_normalized)
		# MARIO INI - Bias por cada clase
		# db = float(error.mean().mean())
		db = error.mean()
		# MARIO FIN
		list_weights -= lr * dw
		bias -= lr * db

	z = data_normalized.dot(list_weights) + bias
	pred = expit(z)
	error = pred - labels
	mse_error = float((error ** 2).mean().mean())

	weights_serializable = list_weights.astype(float).to_dict()
	std_serializable = {
		col: {"Mean": float(df_describe[col]["Mean"]), "Std": float(df_describe[col]["Std"])}
		for col in data_normalized.columns
	}

	json_data = {
		"weights": weights_serializable,
		# MARIO INI - Bias por cada clase (serializar como dict)
		# "bias": float(bias),
		"bias": bias.to_dict(),
		# MARIO FIN
		"mse_error": mse_error,
		"std": std_serializable,
		"columns": data_normalized.columns.to_list()
	}

	out_path = os.path.join(os.path.dirname(__file__), "model.json")
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(json_data, f, indent=2, ensure_ascii=False)