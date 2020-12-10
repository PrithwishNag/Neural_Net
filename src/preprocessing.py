import numpy as np, random
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

class Process:
	def load(self):
		# loads the dataset --> "LBW_Dataset.csv"
		f = open("../data/LBW_Dataset.csv", "r");
		raw = [];
		# storing rows in list raw 
		for line in f.readlines():
			# row seprating with ","
			raw.append(line.strip().split(","))
		raw = raw[1:];
		return raw;

	def fill_median(self, raw):
		# filling empty records with median
		for c in range(len(raw[0])):
			t = [];
			for r in range(len(raw)):
				if raw[r][c] != '':
					raw[r][c] = float(raw[r][c]);
					# collecting all non empty data
					t.append(raw[r][c]);

			# finding median
			t.sort();
			n = len(t);
			if(n%2): med = t[int(n/2)];
			else: med = (t[int((n-1)/2)] + t[int((n+1)/2)])/2;

			# replacing non-empty data with calculated median
			for r in range(len(raw)):
				if raw[r][c] == '':
					raw[r][c] = med;

		return np.array(raw);

	def quantile(self, column, percentile):
		# calculating quantile of given percentile
		t = np.sort(column);
		return t[round(percentile*(t.shape[0] + 1))];
	
	def remove_outliers(self, col, x):
		# removes outliers of a particular column w.r.t interquatile range (iqr)
		arr = np.empty((0, x.shape[1]), int);
		# 25th quantile --> q1
		q1 = self.quantile(x[:, col], 0.25);
		# 75th quantile --> q2
		q3 = self.quantile(x[:, col], 0.75);
		# finding iqr
		iqr = q3 - q1;
		# finding lower and upper threshold
		l1 = q1 - 2.5*iqr;
		l2 = q3 + 2.5*iqr;
		# keeping record with non-outlier row
		for r in range(x.shape[0]):
			if x[r][col] >= l1 and x[r][col] <= l2:
				arr = np.append(arr, np.array([x[r, :]]), 0);
		return arr;

if __name__ == '__main__':
	# creating Process object for preprocessing
	preprocess = Process();
	# loads the data to variable data
	data = preprocess.load();
	# empty value filled with median
	data = preprocess.fill_median(data);
	# removes outliers for blood pressure column 
	data = preprocess.remove_outliers(6, data);
	
	# Min Max Scaling all columns --> x = (x-min)/(max-min)
	scaler = MinMaxScaler();
	data = scaler.fit_transform(data);

	# saving the cleaned data in "clean_data.csv"
	np.savetxt("../data/clean_data.csv", data, delimiter=",");


