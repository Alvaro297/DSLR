from pandas import DataFrame
import pandas as pd
from scipy.special import expit
import json
import os
import numpy as np
import random

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
	for columns in data_normalized.columns:
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

def _check_divergence(loss, prev_loss, iteration):
	if pd.isna(loss) or pd.isna(prev_loss):
		print(f"Warning: Loss NaN detectado en iteración {iteration}")
		return True
	if prev_loss != float('inf') and loss > prev_loss * 1.1:
		print(f"Warning: Divergencia detectada en iteración {iteration} (loss: {loss:.6f}, prev: {prev_loss:.6f})")
		return True
	return False

def _save_model(list_weights, bias, df_describe, data_normalized, labels, mse_error):
	weights_serializable = list_weights.astype(float).to_dict()
	std_serializable = {}
	for col in data_normalized.columns:
		try:
			std_serializable[col] = {
				"Mean": float(df_describe[col]["Mean"]),
				"Std": float(df_describe[col]["Std"])
			}
		except (KeyError, TypeError):
			std_serializable[col] = {"Mean": 0.0, "Std": 1.0}

	json_data = {
		"weights": weights_serializable,
		"bias": bias.to_dict(),
		"mse_error": mse_error,
		"std": std_serializable,
		"columns": data_normalized.columns.to_list()
	}

	out_path = os.path.join(os.path.dirname(__file__), "model.json")
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(json_data, f, indent=2, ensure_ascii=False)

def _batch_gradient_descent(data_normalized, labels, lr, max_iter, patience):
	bias = pd.Series(0.0, index=labels.columns)
	list_weights = pd.DataFrame(0, index=data_normalized.columns, columns=labels.columns)
	prev_loss = float('inf')
	patience_counter = 0
	
	for iteration in range(max_iter):
		z = data_normalized.dot(list_weights) + bias
		pred = expit(z)
		error = pred - labels
		loss = float((error ** 2).mean().mean())
		
		if _check_divergence(loss, prev_loss, iteration):
			break
		
		if loss < prev_loss - 1e-7:
			prev_loss = loss
			patience_counter = 0
		else:
			patience_counter += 1
		
		if patience_counter >= patience:
			break
		
		dw = data_normalized.T.dot(error) / len(data_normalized)
		db = error.mean()
		list_weights -= lr * dw
		bias -= lr * db
	
	return list_weights, bias

def _stochastic_gradient_descent(data_normalized, labels, lr, max_iter, patience):
	bias = pd.Series(0.0, index=labels.columns)
	list_weights = pd.DataFrame(0, index=data_normalized.columns, columns=labels.columns)
	prev_loss = float('inf')
	patience_counter = 0
	
	X = data_normalized.values.copy()
	Y = labels.values.copy()
	W = list_weights.values.copy().astype(float)
	b = bias.values.copy().astype(float)
	n_samples = X.shape[0]
	indices = np.arange(n_samples)
	
	for iteration in range(max_iter):
		np.random.shuffle(indices)
		for idx in indices:
			x_i = X[idx:idx+1]
			y_i = Y[idx:idx+1]
			z = x_i.dot(W) + b
			pred = expit(z)
			error = pred - y_i
			dw = x_i.T.dot(error)
			db = error.flatten()
			W -= lr * dw
			b -= lr * db
		
		z = X.dot(W) + b
		pred = expit(z)
		error = pred - Y
		loss = float(np.mean(error ** 2))
		
		if _check_divergence(loss, prev_loss, iteration):
			break
		
		if loss < prev_loss - 1e-7:
			prev_loss = loss
			patience_counter = 0
		else:
			patience_counter += 1
		
		if patience_counter >= patience:
			break
	
	list_weights = pd.DataFrame(W, index=data_normalized.columns, columns=labels.columns)
	bias = pd.Series(b, index=labels.columns)
	return list_weights, bias

def _mini_batch_gradient_descent(data_normalized, labels, lr, max_iter, patience, batch_size=32):
	bias = pd.Series(0.0, index=labels.columns)
	list_weights = pd.DataFrame(0, index=data_normalized.columns, columns=labels.columns)
	prev_loss = float('inf')
	patience_counter = 0
	
	X = data_normalized.values.copy()
	Y = labels.values.copy()
	W = list_weights.values.copy().astype(float)
	b = bias.values.copy().astype(float)
	n_samples = X.shape[0]
	indices = np.arange(n_samples)
	
	for iteration in range(max_iter):
		np.random.shuffle(indices)
		for start in range(0, n_samples, batch_size):
			end = min(start + batch_size, n_samples)
			batch_idx = indices[start:end]
			x_batch = X[batch_idx]
			y_batch = Y[batch_idx]
			z = x_batch.dot(W) + b
			pred = expit(z)
			error = pred - y_batch
			dw = x_batch.T.dot(error) / len(x_batch)
			db = error.mean(axis=0)
			W -= lr * dw
			b -= lr * db
		
		z = X.dot(W) + b
		pred = expit(z)
		error = pred - Y
		loss = float(np.mean(error ** 2))
		
		if _check_divergence(loss, prev_loss, iteration):
			break
		
		if loss < prev_loss - 1e-7:
			prev_loss = loss
			patience_counter = 0
		else:
			patience_counter += 1
		
		if patience_counter >= patience:
			break
	
	list_weights = pd.DataFrame(W, index=data_normalized.columns, columns=labels.columns)
	bias = pd.Series(b, index=labels.columns)
	return list_weights, bias

def _adam_optimizer(data_normalized, labels, lr, max_iter, patience, beta1=0.9, beta2=0.999, epsilon=1e-8):
	bias = pd.Series(0.0, index=labels.columns)
	list_weights = pd.DataFrame(0, index=data_normalized.columns, columns=labels.columns)
	
	X = data_normalized.values.copy()
	Y = labels.values.copy()
	W = list_weights.values.copy().astype(float)
	b = bias.values.copy().astype(float)
	
	m_w = np.zeros_like(W, dtype=float)
	v_w = np.zeros_like(W, dtype=float)
	m_b = np.zeros_like(b, dtype=float)
	v_b = np.zeros_like(b, dtype=float)
	
	prev_loss = float('inf')
	patience_counter = 0
	
	for iteration in range(1, max_iter + 1):
		z = X.dot(W) + b
		pred = expit(z)
		error = pred - Y
		loss = float(np.mean(error ** 2))
		
		if _check_divergence(loss, prev_loss, iteration):
			break
		
		if loss < prev_loss - 1e-7:
			prev_loss = loss
			patience_counter = 0
		else:
			patience_counter += 1
		
		if patience_counter >= patience:
			break
		
		dw = X.T.dot(error) / len(X)
		db = error.mean(axis=0)
		
		m_w = beta1 * m_w + (1 - beta1) * dw
		v_w = beta2 * v_w + (1 - beta2) * (dw ** 2)
		m_b = beta1 * m_b + (1 - beta1) * db
		v_b = beta2 * v_b + (1 - beta2) * (db ** 2)
		
		m_w_hat = m_w / (1 - beta1 ** iteration)
		v_w_hat = v_w / (1 - beta2 ** iteration)
		m_b_hat = m_b / (1 - beta1 ** iteration)
		v_b_hat = v_b / (1 - beta2 ** iteration)
		
		W -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
		b -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
	
	list_weights = pd.DataFrame(W, index=data_normalized.columns, columns=labels.columns)
	bias = pd.Series(b, index=labels.columns)
	return list_weights, bias

def my_gradient_descent(data: DataFrame, df_describe: DataFrame, df_all: DataFrame, algorithm: str = "batch"):
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
	
	if len(data.columns) == 0:
		print(f"Error: No hay columnas en el DataFrame de datos")
		return
	
	valid_algorithms = ["batch", "sgd", "mini_batch", "adam"]
	if algorithm not in valid_algorithms:
		print(f"Error: Algoritmo '{algorithm}' no válido. Opciones: {valid_algorithms}")
		return
	
	data_normalized = my_normalization(data, df_describe)
	labels = create_labels(df_all)
	labels = labels.reindex(index=data_normalized.index, fill_value=0)
	
	lr = 0.05
	max_iter = 5000
	patience = 100
	
	print(f"Usando algoritmo: {algorithm}")
	
	if algorithm == "batch":
		list_weights, bias = _batch_gradient_descent(data_normalized, labels, lr, max_iter, patience)
	elif algorithm == "sgd":
		list_weights, bias = _stochastic_gradient_descent(data_normalized, labels, lr, max_iter, patience)
	elif algorithm == "mini_batch":
		list_weights, bias = _mini_batch_gradient_descent(data_normalized, labels, lr, max_iter, patience)
	elif algorithm == "adam":
		list_weights, bias = _adam_optimizer(data_normalized, labels, lr, max_iter, patience)
	
	z = data_normalized.dot(list_weights) + bias
	pred = expit(z)
	error = pred - labels
	mse_error = float((error ** 2).mean().mean())
	
	_save_model(list_weights, bias, df_describe, data_normalized, labels, mse_error)