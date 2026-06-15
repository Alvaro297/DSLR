#!/usr/bin/env python3

import argparse
from pandas import DataFrame
import pandas as pd
from gradient_descent import my_gradient_descent
from describe import describe_dataset
from scatter_plot import my_scatter_plot
from histogram import historigram
from pair_plot import my_pair_plot

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

	if args.verbose:
		print("Modo verbose activado")
	dataset_pandas: DataFrame = pd.read_csv(args.csv)
	df_describe = describe_dataset(dataset_pandas)
	df_ravenclaw = dataset_pandas[dataset_pandas['Hogwarts House'] == 'Ravenclaw']
	df_slytherin = dataset_pandas[dataset_pandas['Hogwarts House'] == 'Slytherin']
	df_gryffindor = dataset_pandas[dataset_pandas['Hogwarts House'] == 'Gryffindor']
	df_hufflepuff = dataset_pandas[dataset_pandas['Hogwarts House'] == 'Hufflepuff']
	datasets = [df_ravenclaw, df_slytherin, df_gryffindor, df_hufflepuff]
	historigram(datasets)
	my_scatter_plot(datasets)
	data = my_pair_plot(dataset_pandas)
	my_gradient_descent(data, df_describe, dataset_pandas)

if __name__ == "__main__":
	main()