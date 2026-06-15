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
	print(df_describe.columns)
	print("El data frame")
	print(data_normalized.columns)
	for columns in data_normalized.columns:
		print(columns)
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
	data_normalized = my_normalization(data, df_describe)
	lr = 0.05
	bias = 1.0
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
		db = float(error.mean().mean())
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
		"bias": float(bias),
		"mse_error": mse_error,
		"std": std_serializable,
		"columns": data_normalized.columns.to_list()
	}

	out_path = os.path.join(os.path.dirname(__file__), "model.json")
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(json_data, f, indent=2, ensure_ascii=False)