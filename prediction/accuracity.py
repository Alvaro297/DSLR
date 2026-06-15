import pandas as pd
import numpy as np
import os


def main():
	base_dir = os.path.dirname(__file__)
	csv_path = os.path.join(base_dir, "..", "datasets", "dataset_train.csv")

	df = pd.read_csv(csv_path)
	p_train = 0.70 # Porcentaje de train.

	df['is_train'] = np.random.uniform(0, 1, len(df)) <= p_train
	train, test = df[df['is_train']==True], df[df['is_train']==False]
	df = df.drop(columns=['is_train'])
	train = train.drop(columns=['is_train'])
	test = test.drop(columns=['is_train'])

	print("Ejemplos usados para entrenar: ", len(train))
	print("Ejemplos usados para test: ", len(test))

	train_path = os.path.join(base_dir, "train_split.csv")
	test_path = os.path.join(base_dir, "validation_split.csv")

	train.to_csv(train_path)
	test.to_csv(test_path)
	




if __name__ == "__main__":
	main()