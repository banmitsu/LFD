import numpy as np
import random as rd

import matplotlib.pyplot as plt
import matplotlib.lines as lines
from numpy.linalg import inv

class HighlightSelected(lines.VertexSelector):
    def __init__(self, line, fmt='ro', **kwargs):
        lines.VertexSelector.__init__(self, line)
        self.markers, = self.axes.plot([], [], fmt, **kwargs)
    def process_selected(self, ind, xs, ys):
        self.markers.set_data(xs, ys)
        self.canvas.draw()

def choose_2D_target_function(x1_range, x2_range):
	x1_min, x1_max = x1_range
	x2_min, x2_max = x2_range
	# pick two points, calculate slope	 
	X1 = (rd.uniform(x1_min, x1_max), rd.uniform(x1_min, x1_max))
	X2 = (rd.uniform(x2_min, x2_max), rd.uniform(x2_min, x2_max))
	slope = (X2[1]-X2[0])/(X1[1]-X1[0])
	return (X1[0], X2[0]), (X1[1], X2[1]), slope

def plot_target_function_line(slope, x, x1_range, x2_range):
	x1_min, x1_max = x1_range
	x2_min, x2_max = x2_range
	intersect_x_max = slope*(x1_max-x[0])+x[1]
	intersect_x_min = slope*(x1_min-x[0])+x[1]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlim(x1_min, x1_max)
	ax.set_ylim(x2_min, x2_max)
	line, = plt.plot([x1_max, x1_min], [intersect_x_max, intersect_x_min], 'b-', picker=5)
	selector = HighlightSelected(line)
	fig.set_size_inches(6, 6)
	plt.draw()

def plot_data(X1, X2, Y):
    plt.scatter(X1[Y >= 0], X2[Y >= 0], s = 80, c = 'b', marker = "o")
    plt.scatter(X1[Y <  0], X2[Y  < 0], s = 80, c = 'r', marker = "^")
    plt.draw()

def draw_line(w, x1_range):
	x1_min, x1_max = x1_range
	#slope = w[1]/w[2]
	if w[2] != 0:
		intersect_x1_max = (-w[0]-(w[1]*x1_max))/w[2]
		intersect_x1_min = (-w[0]-(w[1]*x1_min))/w[2]
		line, = plt.plot([x1_max, x1_min], [intersect_x1_max, intersect_x1_min], 'r-', picker=5)
		plt.draw()
	elif w[1] != 0:
		line, = plt.plot([ (w[0]/w[1]), (w[0]/w[1])], [1, -1], 'r-', picker=5)
		plt.draw()
	else:
		print "ERROR, Invalid w1 and w2."

def generate_data(N):
	# generate train data according to the target function
	# target function: (x_1)^2 + (x_2)^2 - 0.6
	
	#fig = plt.figure()
	#ax = fig.add_subplot(111)
	#ax.set_xlim(-1, 1)
	#ax.set_ylim(-1, 1)
	#circle = plt.Circle((0, 0), np.sqrt(0.6), color='b', fill=False)
	#ax.add_artist(circle)
	noise = np.array([rd.uniform(0, 1) for i in range(N)])
	noise[noise>=0.1] = 1
	noise[noise <0.1] = -1

	X0 = np.array([(-0.6) for i in range(N)])
	X1 = np.array([rd.uniform(-1, 1) for i in range(N)])
	X2 = np.array([rd.uniform(-1, 1) for i in range(N)])
	Y  = np.array([np.sign(X1[i]*X1[i]+X2[i]*X2[i]+X0[i]) for i in range(N)])
	Y  = np.multiply(Y, noise)
	#plot_data(X1, X2, Y)
	return X0, X1, X2, Y
	

def linear_regression(X, Y):
	N = len(Y)

	# nonlinear transformation:
	X0 = np.array([1 for i in range(N)])
	X1 = X[1]
	X2 = X[2]
	X3 = np.multiply(X[1], X[2])
	X4 = np.multiply(X[1], X[1])
	X5 = np.multiply(X[2], X[2])

	X_T = np.concatenate(([X0], [X1], [X2], [X3], [X4], [X5]), axis=0)
	X_dagger = np.dot( inv(np.dot(X_T, X_T.T)), X_T ) #pseudo-inverse
	W = np.dot( X_dagger, Y.reshape(N,1))

	square_err = np.sum((np.dot(X_T.T, W) - Y)*(np.dot(X_T.T, W) - Y))/N
	y = np.sign(np.dot(X_T.T, W))
	y.shape = (N,)
	acc = sum(y == Y)
	E_in = (N-acc)/float(N)

	#draw_line((W[0], W[1], W[2]), (-1, 1))
	#plt.show()
	return W, E_in

def estimate_out_of_sample_error(X, Y, W):
	N = len(Y)

	# nonlinear transformation:
	X0 = np.array([1 for i in range(N)])
	X1 = X[1]
	X2 = X[2]
	X3 = np.multiply(X[1], X[2])
	X4 = np.multiply(X[1], X[1])
	X5 = np.multiply(X[2], X[2])
	X_T = np.concatenate(([X0], [X1], [X2], [X3], [X4], [X5]), axis=0)

	square_err = np.sum((np.dot(X_T.T, W) - Y)*(np.dot(X_T.T, W) - Y))/N
	y = np.sign(np.dot(X_T.T, W))
	y.shape = (N,)
	acc = sum(y == Y)
	E_out = (N-acc)/float(N)

	#draw_line((W[0], W[1], W[2]), (-1, 1))
	#plt.show()
	return E_out

def main():
	##########################################
	# linear regression
	##########################################
	
	N = 1000
	runs = 1000
	ERR_in = np.array([0 for i in range(runs)])
	ERR_out= np.array([0 for i in range(runs)])

	for run in range(runs):
		print("Iteration: ", run)
		X0, X1, X2, Y = generate_data(N)
		W, E_in = linear_regression((X0, X1, X2), Y)

		X0, X1, X2, Y = generate_data(N)
		E_out 	= estimate_out_of_sample_error((X0, X1, X2), Y, W)

		ERR_in[run] = E_in
		ERR_out[run]= E_out
	print np.mean(E_in)
	print np.mean(E_out)
	print W

if __name__ == '__main__':
	main()