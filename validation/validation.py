import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score

def main():
	base_dir = os.path.dirname(__file__)
	# MARIO INI - Cambiar a houses.csv
	# predictions_path = os.path.join(base_dir, "..", "predictions.csv")
	predictions_path = os.path.join(base_dir, "..", "houses.csv")
	# MARIO FIN
	validation_path = os.path.join(base_dir, "..", "prediction", "validation_split.csv")

	validation = pd.read_csv(validation_path)
	validation = validation.reset_index(drop=True)

	prediction = pd.read_csv(predictions_path)

	y_real = validation["Hogwarts House"]
	y_pred = prediction["Hogwarts House"]
	accuracy = accuracy_score(y_real, y_pred)
	print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
	main()