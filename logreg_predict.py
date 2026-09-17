import os
import pandas as pd
import json
import sys
from pandas import DataFrame
from scipy.special import expit
import argparse


def normalizacion(df_result: DataFrame, json_trianed: dict) -> DataFrame:
	df_norm = df_result.copy()
	for col in df_norm.columns:
		stats = json_trianed.get("std", {}).get(col)
		if stats is None:
			continue
		mean = stats.get("Mean", 0.0)
		std = stats.get("Std", 1.0)
		df_norm[col] = (df_norm[col] - mean) / std if std else (df_norm[col] - mean)
	return df_norm


def main():
	parser = argparse.ArgumentParser(description="Genera predicciones usando el modelo entrenado")
	parser.add_argument("dataset", type=str, help="Ruta al archivo CSV de test")
	parser.add_argument(
		"-v", "--verbose",
		action="store_true",
		help="activar salida detallada"
	)
	# MARIO INI - Aceptar modelo como parametro opcional
	parser.add_argument(
		"--model",
		type=str,
		default=None,
		help="Ruta al archivo del modelo (default: model.json)"
	)
	# MARIO FIN
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
	
	base_dir = os.path.dirname(__file__)
	# MARIO INI - Usar modelo especificado o default
	if args.model:
		model_path = args.model
	else:
		model_path = os.path.join(base_dir, "model.json")
	# MARIO FIN
	
	# MARIO INI - Validaciones del modelo
	if not os.path.exists(model_path):
		print(f"Error: El archivo del modelo '{model_path}' no existe")
		print("Primero debes entrenar el modelo con logreg_train.py")
		sys.exit(1)
	
	if not os.path.isfile(model_path):
		print(f"Error: '{model_path}' no es un archivo")
		sys.exit(1)
	
	if os.path.getsize(model_path) == 0:
		print(f"Error: El archivo del modelo '{model_path}' está vacío")
		sys.exit(1)
	# MARIO FIN
	
	weights_path = os.path.join(base_dir, "weights.csv")

	try:
		df = pd.read_csv(dataset_path)
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
	if df.empty:
		print(f"Error: El dataset está vacío")
		sys.exit(1)
	
	if len(df) == 0:
		print(f"Error: El dataset no tiene filas")
		sys.exit(1)
	# MARIO FIN

	try:
		with open(model_path, "r") as archivo:
			json_trained = json.load(archivo)
	except json.JSONDecodeError as e:
		print(f"Error: El archivo del modelo no es un JSON válido: {e}")
		sys.exit(1)
	except Exception as e:
		print(f"Error al leer el archivo del modelo: {e}")
		sys.exit(1)
	
	# MARIO INI - Validaciones del modelo cargado
	if "weights" not in json_trained:
		print(f"Error: El archivo del modelo no contiene 'weights'")
		sys.exit(1)
	
	if "columns" not in json_trained:
		print(f"Error: El archivo del modelo no contiene 'columns'")
		sys.exit(1)
	
	if "bias" not in json_trained:
		print(f"Error: El archivo del modelo no contiene 'bias'")
		sys.exit(1)
	# MARIO FIN

	weights: DataFrame = pd.DataFrame(json_trained["weights"])
	weights.to_csv(weights_path)

	valid_list: list = json_trained["columns"]
	
	# MARIO INI - Validaciones de columnas
	missing_cols = [col for col in valid_list if col not in df.columns]
	if missing_cols:
		print(f"Error: Faltan columnas requeridas en el dataset: {missing_cols}")
		sys.exit(1)
	# MARIO FIN

	df_resultado: DataFrame = df[valid_list].copy()

	for columna, stats in json_trained["std"].items():
		if columna in df_resultado.columns:
			fill_value = stats["Mean"]
			df_resultado[columna] = df_resultado[columna].fillna(fill_value)
	
	df_normalized = normalizacion(df_resultado, json_trained)
	weights = weights.reindex(df_normalized.columns).fillna(0.0)

	# MARIO INI - Bias por cada clase (cargar como Series)
	# bias = json_trained["bias"]
	bias_data = json_trained["bias"]
	if isinstance(bias_data, dict):
		bias = pd.Series(bias_data)
	else:
		bias = bias_data
	# MARIO FIN
	
	z = df_normalized.dot(weights) + bias
	
	# MARIO INI - Debug solo en modo verbose
	if args.verbose:
		print("Predicciones (z):")
		print(z.head())
		print(f"\nForma de z: {z.shape}")
		print(f"Columnas de z (clases): {z.columns.tolist()}")
	# MARIO FIN
	
	pred = expit(z)
	max_value = pred.idxmax(axis=1)
	# MARIO INI - Eliminar columna Probability para cumplir con el subject
	# max_prob = pred.max(axis=1)
	df_predictions = pd.DataFrame({
		"Hogwarts House": max_value,
		# "Probability": max_prob,
	})
	# MARIO FIN
	
	# MARIO INI - Cambiar nombre a houses.csv
	# predictions_path = os.path.join(base_dir, "predictions.csv")
	houses_path = os.path.join(base_dir, "houses.csv")
	# MARIO FIN
	
	df_predictions = df_predictions.reset_index().rename(columns={"index": "Index"})
	# MARIO INI - Guardar como houses.csv
	# df_predictions.to_csv(predictions_path, index=False)
	df_predictions.to_csv(houses_path, index=False)
	print(f"Predicciones guardadas en {houses_path}")
	# MARIO FIN

if __name__ == "__main__":
	main()