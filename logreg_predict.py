import os
import pandas as pd
import json
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
	parser = argparse.ArgumentParser(description="Programa principal de dslr")
	parser.add_argument(
		"-v", "--verbose",
		action="store_true",
		help="activar salida detallada"
	)
	parser.add_argument(
		"--csv",
		type=str,
		required=True,
		help="Ruta al archivo CSV"
	)
	args = parser.parse_args()
	base_dir = os.path.dirname(__file__)
	model_path = os.path.join(base_dir, "model.json")
	weights_path = os.path.join(base_dir, "weights.csv")

	df = pd.read_csv(args.csv)

	with open(model_path, "r") as archivo:
		json_trained = json.load(archivo)

	weights: DataFrame = pd.DataFrame(json_trained["weights"])
	weights.to_csv(weights_path)

	valid_list: list = json_trained["columns"]

	df_resultado: DataFrame = df[valid_list].copy()

	for columna, stats in json_trained["std"].items():
		if columna in df_resultado.columns:
			fill_value = stats["Mean"]
			df_resultado[columna] = df_resultado[columna].fillna(fill_value)
	
	df_normalized = normalizacion(df_resultado, json_trained)
	weights = weights.reindex(df_normalized.columns).fillna(0.0)

	bias = json_trained["bias"]
	z = df_normalized.dot(weights) + bias
	print("Predicciones (z):")
	print(z.head())
	print(f"\nForma de z: {z.shape}")
	print(f"Columnas de z (clases): {z.columns.tolist()}")
	pred = expit(z)
	max_value = pred.idxmax(axis=1)
	max_prob = pred.max(axis=1)
	df_predictions = pd.DataFrame({
		"Hogwarts House": max_value,
		"Probability": max_prob,
	})
	predictions_path = os.path.join(base_dir, "predictions.csv")
	df_predictions = df_predictions.reset_index().rename(columns={"index": "Index"})
	df_predictions.to_csv(predictions_path, index=False)

if __name__ == "__main__":
	main()