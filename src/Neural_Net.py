import numpy as np, random
from sklearn.model_selection import train_test_split

# Global list collection of all "layer" objects
layers = [];
# Global list collection of all neuron count in each layer
neurons_list = [];
# Seed for random weights
np.random.seed(0);

# Class for the activation functions
class Activation_fn:
	def leaky_Relu(self, x, deriv=False):
		# 		|x for x >= 0
		# f(x)	|
		# 		|0.01*x for x < 0  
		c = 0.01;
		if deriv:
			if x >= 0: return 1;
			else: return c;
		if x >= 0: return x;
		return c*x;

	def sigmoid(self, x, deriv=False):
		# f(x) = 1 / (1 + e^-x)
		ex = np.exp(-x);
		f = 1 / (1 + ex);
		# df/dx = f * (1 - f)
		df = f * (1 - f);
		if deriv: return df;
		return f;

	def tanh(self, x, deriv=False):
		# f(x) = (e^x - e^-x)/(e^x + e^-x)
		exp = np.exp(x);
		exn = np.exp(-x);
		th = (exp - exn)/(exp + exn);
		# df/dx = (1 - f^2)
		if deriv: return (1-pow(th,2));
		return th;
		
class Matrix_Operations:
	def hadamard(self, x, y):
		# computing hadamard product
		# x * y --> for all i, j --> x[i][j] * y[i][j];
		z = x * y;
		return z;

	def transpose(self, mat): 
		# exchanging rows and columns
		return mat.reshape(mat.shape[::-1]);

class Layer:
	def __init__(self, neurons, act_fn):
		# initialising number of neurons
		self.neurons = neurons;
		# Using "Xaviers" initialization for weights 
		self.weights = np.sqrt(2/neurons_list[-1])*np.random.randn(neurons, neurons_list[-1]);
		# initialising bias to 0
		self.bias = np.zeros(neurons).reshape(-1,1);
		# initialising activation function used in the layer
		self.act_fn = act_fn;
		# z = w * a - b
		self.z = None;
		# error vector delta(e_L) initialised to None
		self.delta = None;

	def add(self):
		# appends "layer" object to the Global layer list
		layers.append(self);
		# appends number of neurons in present layer to the global list
		neurons_list.append(self.neurons);
		# returning the object
		return self;

	def activated_sum(self, activations):
		# Computing Weigted sum
		# z = w * a - b
		self.z = np.dot(self.weights, activations) - self.bias;
		
		# self.act_fn(activation function in string) to actual function stored in act_fn
		act_obj = Activation_fn();
		act_fn = getattr(act_obj, self.act_fn);

		# Passing each value of z through Activation function
		a = np.zeros(self.z.shape);
		for i in range(self.z.shape[0]):
			a[i][0] = act_fn(self.z[i][0]);
		# returning the activation a = act_fn(w*a-b)
		return a;

class Propagation:
	def delta(self, wl, el, act_fn, zl):
		mat_obj = Matrix_Operations();
		# t1 = ((w_l+1)_T e_L+1);
		t1 = np.dot(mat_obj.transpose(wl), el);
		
		# act_fn(activation function in string) to actual function stored in act_fn
		act_obj = Activation_fn();
		act_fn = getattr(act_obj, act_fn);

		# t2 = derv_act_fn(z_l)
		t2 = np.zeros(zl.shape);
		for i in range(zl.shape[0]):
			# for each value in z vector passing throught derivative of activation function
			t2[i][0] = act_fn(zl[i][0], True);

		# e_L = hadamard product of t1 and t2
		return mat_obj.hadamard(t1, t2);

	def update_weights(self, activations, lr):
		mat_obj = Matrix_Operations();
		for i, l in enumerate(layers):
			# computing the gradient which is the cange in weights to recover the loss
			grad = np.dot(l.delta, mat_obj.transpose(activations[i]));
			# updating weights to product of learning rate and weight gradient 
			l.weights += lr * grad;
			# updating bias to the delta(e_L) computed
			l.bias += lr * l.delta;

	def feed_forward(self, a1):
		# computing activation[l] = act(w*activation[l-1]-b) for each layer and propagating forward
		activations = [a1];
		for layer in layers:
			activations.append(layer.activated_sum(activations[-1]));
		# return all the activations calulated in a list
		return activations;

	def back_propagate(self, activations, y_true):
		# computing the mse loss here
		# derivative of (y_true - a)^2 is y_true - a in terms of power, where a is last layer activation hence...
		layers[-1].delta = y_true - activations[-1]; 
		for l in range(len(layers)-2, -1, -1):
			# derivative of loss w.r.t weights give the below e_L*a[l-1] for weights and e_L for bias
			# e_L is delta specified in the Layer class
			# e_L = ((w_l+1)_T e_L+1).(derv_act_fn)(z_l)
			layers[l].delta = self.delta(layers[l+1].weights, layers[l+1].delta, layers[l+1].act_fn, layers[l].z);

	def trip(self, a1, y_true, lr):
		# Front propagation
		activations = self.feed_forward(a1);
		# Back propagation
		self.back_propagate(activations, y_true);
		# Updating the weights
		self.update_weights(activations, lr);

class NN:
	def __init__(self):
		self.model = None;
		# intialising Learning rate as 0.1
		self.learning_rate = 0.1;
		# intialising Epochs as 800 
		self.epochs = 800;

	def fit(self, X, Y):
		# putting input layer neurons in neurons list
		neurons_list.append(X.shape[1]);
		
		# initialising Propagation class object
		self.model = Propagation();
		
		# Generalised Model can add any number of hidden and output layers
		# just add Layer(number of neurons in that layer, "activation function").add();

		# Hidden Layer 1 with neurons 16 and activation function "sigmoid" 
		Layer(16, "sigmoid").add();
		# Hidden Layer 2 with neurons 16 and activation function "sigmoid" 
		Layer(16, "sigmoid").add();
		# Output Layer 1 with neurons 1 and activation function "sigmoid"
		Layer(1, "sigmoid").add();

		# fitting the model for each epoch
		for e in range(self.epochs):
			# for every sample propagating and updating
			for i in range(len(X)):
				self.model.trip(X[i].reshape(-1,1), Y[i].reshape(-1,1), self.learning_rate);

	def predict(self, X):
		# just feed forwarding the X matrix to get desired binary output
		yhat = [];
		for x in X:
			# saving binary output
			yhat.append(self.model.feed_forward(x.reshape(-1,1))[-1][0][0]);
		return np.array(yhat);

	def CM(self, y_test, y_test_obs):
		for i in range(len(y_test_obs)):
			if(y_test_obs[i]>0.6):
				y_test_obs[i]=1
			else:
				y_test_obs[i]=0
		
		cm=[[0,0],[0,0]]
		fp=0
		fn=0
		tp=0
		tn=0

		for i in range(len(y_test)):
			if(y_test[i]==1 and y_test_obs[i]==1):
				tp=tp+1
			if(y_test[i]==0 and y_test_obs[i]==0):
				tn=tn+1
			if(y_test[i]==1 and y_test_obs[i]==0):
				fp=fp+1
			if(y_test[i]==0 and y_test_obs[i]==1):
				fn=fn+1

		cm[0][0]=tn
		cm[0][1]=fp
		cm[1][0]=fn
		cm[1][1]=tp

		a = (tn+tp)/(tp+tn+fp+fn);
		p = tp/(tp+fp)
		r = tp/(tp+fn)
		f1 = (2*p*r)/(p+r)
		
		print("Confusion Matrix : ")
		print(cm)
		print()
		print(f"Accuracy \t: {a}")
		print(f"Precision \t: {p}")
		print(f"Recall \t\t: {r}")
		print(f"F1 SCORE \t: {f1}")
		return a;

if __name__ == '__main__':
	# Opening the clean data
	f = open("../data/clean_data.csv", "r");
	data = [];
	# storing rows in list raw 
	for line in f.readlines():
		# row seprating with ","
		data.append(list(map(float, line.strip().split(","))))
	data = np.array(data);

	# x data split from full data
	x = data[:,:-1];
	# y data split from full data
	y = data[:, -1];

	# sklearn used for spliting the data
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0);	

	# main fitting and predicting using own created NN class
	ob = NN();
	# fitting the model(modifying weights)
	print("Fitting in progress...")
	ob.fit(X_train, y_train);

	# confusion matrix for Training data
	print("_____Training_____");
	a1 = ob.CM(y_train, ob.predict(X_train));
	print()

	# confusion matrix for Testing data
	print("_____Testing_____");
	a2 = ob.CM(y_test, ob.predict(X_test));