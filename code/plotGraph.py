import re
import numpy as np
import matplotlib.pyplot as plt

import re
import numpy as np
import matplotlib.pyplot as plt

def plotHistoryGraph(strData='nothing', epoch=100):

	# parsing point : loss: [6글자], accuracy: [6글자], val_loss: [6글자], val_accuracy: [6글자]
	epoch_data = strData.split(sep="Epoch")
	nums = [re.findall(r' [0-9]+[.][0-9]+', i) for i in epoch_data[1:]] # loss, accuracy, val_loss, val_accuracy
	
	# loss, accuracy, val_loss, val_accuracy
	loss = np.array([ float(i[0].strip()) for i in nums ])
	accuracy = np.array([ float(i[1].strip()) for i in nums ])
	val_loss = np.array([ float(i[2].strip()) for i in nums ])
	val_accuracy = np.array([ float(i[3].strip()) for i in nums ])

	# loss 기준점 : 0.8
	# accuracy 기준점 : 0.8
	dt = 0.01
	t = np.arange(0, epoch)
	fig, axs = plt.subplots(4, 1, figsize=(15,20))

	# Loss
	lines = axs[0].plot(t, loss, t, val_loss)
	plt.setp(lines[0], alpha=0.6)
	plt.setp(lines[1], linewidth=3)
	axs[0].set_xlim(0, epoch-1)
	axs[0].set_xlabel('Epoch')
	axs[0].set_ylim(0, 1.1)
	axs[0].set_ylabel('Training loss and Validation loss')
# 	axs[0].grid(True)

	line = axs[1].plot(t, abs(loss-val_loss)*100)
	plt.setp(line, linewidth=3)
	axs[1].set_xlim(0, epoch-1)
	axs[1].set_xlabel('Epoch')
	axs[1].set_ylim(0, 100)
	axs[1].set_ylabel('Loss Coherence %')
# 	axs[1].grid(True)
# 	cxy, f = axs[1].cohere(loss, val_loss, 1, 0.2)

	# Accuracy
	lines = axs[2].plot(t, accuracy, t, val_accuracy)
	plt.setp(lines[0], alpha=0.6)
	plt.setp(lines[1], linewidth=3)
	axs[2].set_xlim(0, epoch-1)
	axs[2].set_xlabel('Epoch')
	axs[2].set_ylim(0, 1.1)
	axs[2].set_ylabel('Training accuracy and Validation accuracy')
# 	axs[2].grid(True)

# 	cxy2, f2 = axs[3].cohere(accuracy, val_accuracy)
	line = axs[3].plot(t, abs(loss-val_loss)*100)
	plt.setp(line, linewidth=3)
	axs[3].set_xlim(0, epoch-1)
	axs[3].set_xlabel('Epoch')
	axs[3].set_ylim(0, 100)
# 	axs[3].grid(True)
# 	cxy, f = axs[1].cohere(loss, val_loss, 1, 0.2)
	axs[3].set_ylabel('Acc Coherence %')

	fig.tight_layout()
	plt.show()


def plotDiffGraph(strData='', epoch=100):

	# parsing point : loss: [6글자], accuracy: [6글자], val_loss: [6글자], val_accuracy: [6글자]
	epoch_data = strData.split(sep="Epoch")
	nums = [re.findall(r' [0-9]+[.][0-9]+', i) for i in epoch_data[1:]] # loss, accuracy, val_loss, val_accuracy
	
	# loss, accuracy, val_loss, val_accuracy
	loss = np.array([ float(i[0].strip()) for i in nums ])
	accuracy = np.array([ float(i[1].strip()) for i in nums ])
	val_loss = np.array([ float(i[2].strip()) for i in nums ])
	val_accuracy = np.array([ float(i[3].strip()) for i in nums ])

	# loss 기준점 : 0.6 (사람의 오류율)
	# accuracy 기준점 : 0.8
	t = np.arange(0, epoch)
	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15,6))

	ax1.fill_between(t, 0.6, val_loss)
	ax1.set_xlim(0, epoch-1)
	ax1.set_ylim(0, max(val_accuracy))
	ax1.set_ylabel('Between 1 to 0')
	ax1.set_xlabel('Epoch')

	ax2.fill_between(t, 0.8, val_loss)
	ax1.set_xlim(0, epoch-1)
	ax1.set_ylim(0, max(val_accuracy))
	ax2.set_ylabel('Between 1 to 0')
	ax2.set_xlabel('Epoch')
    
	fig.tight_layout()
	plt.show()



