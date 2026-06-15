import seaborn as sns
from pandas import DataFrame
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def my_pair_plot(datasets: DataFrame, save_path: str = None, sample_n: int = 500, show: bool = True) -> DataFrame:
	colors = ['blue', 'green', 'red', 'orange']
	data = datasets
	if sample_n and len(datasets) > sample_n:
		data = datasets.sample(sample_n, random_state=0)

	threshold = 0.8
	columnas_numericas = data.select_dtypes(include='number').columns.tolist()
	for col1 in columnas_numericas:
		for col2 in columnas_numericas:
			if col1 == col2 or (col1 == "Index" or col2 == "Index"):
				continue
			c1 = col1
			c2 = col2
			clean = data[[c1, c2]].dropna()
			if len(clean) > 2:
				corr, pval = pearsonr(clean[c1], clean[c2])
				if abs(corr) >= threshold:
					data = data.drop(columns=c2)
					columnas_numericas.remove(col2)

	data = data.drop(columns="Index")
	g = sns.pairplot(data, hue="Hogwarts House", palette=colors)

	if save_path:
		try:
			g.savefig(save_path)
		except Exception:
			pass

	if show:
		plt.show()
	else:
		plt.close()
	return data