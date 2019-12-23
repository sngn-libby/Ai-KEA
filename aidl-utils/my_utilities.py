import os
import pickle
import csv
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import itertools

def plotter(x,y1,y2, ylabel, title):
	"""
	Plots values from the training histrograma (learning curves, etc.)
	"""
	fig = plt.figure()
	ax = plt.subplot(111)
	ax.plot(x,y1)
	ax.plot(x,y2)
	ax.legend(['train','val'])
	plt.xlabel('Number of Epochs')
	plt.ylabel(ylabel)
	plt.title(title)
	plt.grid(True)
	plt.show()

def predictions_decoder(preds, top=5):
	"""
	Takes an array with probabilities, obtains the indices of the highest ones and
	matches them with labels loaded from a .csv file
	"""
	FILENAME = 'labels.csv'
	print("Loading labels from: %s" % FILENAME)

	labels = csv_loader(FILENAME)
				
	results = []
	for pred in preds:
		top_indices = pred.argsort()[-top:][::-1]
		result = [tuple(labels[i]) + (np.around(pred[i],5),) for i in top_indices]
		results.append(result)
	return results[0]

def cnf_matrix_plotter(cm, classes):
	"""
	Plots the confusion matrix
	"""
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
	plt.title('Confusion Matrix')
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	threshold = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
				 horizontalalignment="center",
				 color="white" if cm[i, j] > threshold else "black")

	plt.tight_layout()
	plt.ylabel('True Label')
	plt.xlabel('Predicted Label')
	plt.show()

def csv_saver(filename, data):

	pass

def csv_loader(filename):

	data = []
	with open(filename,'r') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			data.append(row)

	return data	

def pickle_saver(filename, data):
	
	ROOT_DIR = 'plots/'
	FILENAME = filename
	SAVE_DIR = osp.join(ROOT_DIR, FILENAME + ".pickle")
	
	with open(SAVE_DIR, 'wb') as f:
		print("Saving plot data to: %s" % SAVE_DIR)
		pickle.dump(data, f)

	print("...done")

def pickle_loader(filename):
	"""
	Loads serialized (pickled) data structures
	"""
	ROOT_DIR = 'plots/'
	FILENAME = filename

	LOAD_DIR = osp.join(ROOT_DIR, FILENAME + ".pickle")

	with open(LOAD_DIR, 'rb') as f:
		print("Loading plot data from: %s " % LOAD_DIR)
		data = pickle.load(f)
		print("...done")
	
	return data