import numpy as np


def relu(x):
	return np.maximum(x, 0)


def sigmoid(x):
	return np.exp(x) / (1 + np.exp(x))


def generate_data(seed: int, m: int) -> (np.array, np.array, np.array, np.array):
	"""
	generate random simulation data according to the set up in "https://www.ijcai.org/proceedings/2017/0318.pdf"
	@param m: nonzero input
	@param seed: seed
	@return: x_train, x_test, y_train, y_test
	"""
	"""parameters"""
	np.random.seed(seed)
	p = 10000
	n = 8500
	train_size = 1000
	h1 = 50
	h2 = 30
	h3 = 15
	h4 = 10
	sig = np.sqrt(0.5)
	"""generate x"""
	x = np.random.rand(n*p).reshape(n, p) * 2 - 1
	"""neural network forward"""
	w1 = np.concatenate([np.random.randn(m, h1) * sig, np.zeros([p-m, h1])])
	w2 = np.random.randn(h1, h2) * sig
	w3 = np.random.randn(h2, h3) * sig
	w4 = np.random.randn(h3, h4) * sig
	w5 = np.random.randn(h4, 1) * sig
	z1 = np.matmul(x, w1)
	a1 = relu(z1)
	z2 = np.matmul(a1, w2)
	a2 = relu(z2)
	z3 = np.matmul(a2, w3)
	a3 = relu(z3)
	z4 = np.matmul(a3, w4)
	a4 = relu(z4)
	prob = sigmoid(np.matmul(a4, w5))
	"""generate y"""
	y = np.where(prob > 0.5, 1, 0)
	"""flip 5% labels"""
	flip = np.random.choice(range(n), size=int(n*0.05))
	y[flip, :] = 1 - y[flip, :]
	print(f"y has mean {y.mean()}")
	return x[:train_size, :], x[train_size:, :], y[:train_size], y[train_size:]
