import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import CTP_utils_2d
from CTP_config import config
from torch.cuda import amp

################################################################################
################################ Auxiliary functions ###########################
################################################################################
## Optimziation method
def get_optim(optim_method, model, learning_rate, weight_decay):

	if optim_method == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

	elif optim_method == 'sgd':
		# optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0002)
		optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

	elif optim_method == 'RMSprop':
		optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

	else:
		sys.exit('[get_optim] Error: please select a valid optimization method -- exiting')

	return optimizer

## Learning rate decay scheduler
def create_scheduler(optimizer, num_epochs):

	# lr_new = gamma * lr_old if epoch%step_size = 0
	# lr_new = lr_old otherwise
	if config.lr_decay_strategy == 'step':
		scheduler = optim.lr_scheduler.StepLR(optimizer, config.decay_step_size, config.decay_gamma)

	# lr_new = lr_init * lambda(epoch)
	elif config.lr_decay_strategy == 'decay':
		print("decay rate: ", config.decay_rate)
		scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0 / (1.0 + config.decay_rate * epoch))

	# lr_new = lr_init * lambda(epoch)
	elif config.lr_decay_strategy == 'sqrt':
		scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0 / (1.0 + np.sqrt(epoch)))

	# lr_new = gamma * lr_old
	elif config.lr_decay_strategy == 'exp':
		print("decay rate: ", config.decay_gamma)
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.decay_gamma)

	else:
		sys.exit('[create_scheduler] Error: please select a valid learning rate scheduler -- exiting')

	return scheduler

### Loss functions
def get_loss_fn(loss_fn):
	if loss_fn == 'mse':
		loss_function = nn.MSELoss()
	elif loss_fn == 'mae':
		loss_function = nn.L1Loss()
	elif loss_fn == 'huber':
		loss_function = nn.SmoothL1Loss()
	elif loss_fn == 'log_cosh':
		sys.exit('[get_loss_fn] Error: log cosh loss function not implemented yet -- exiting')
	else:
		sys.exit('[get_loss_fn] Error: please select a valid loss function -- exiting')
	return loss_function

### Compute loss function
def compute_loss_2d(data, labels, model, loss_function):
	model.eval() # Set eval mode (used for prediction: turns off dropout, etc.)
	with torch.no_grad():
		y_pred = forward_by_batch(data, model, config.batch_size)
		accuracy = compute_accuracy_2d(y_pred, labels, config.ac_threshold)
		loss_value = loss_function(y_pred, labels)

	model.train() # Set mode back to training
	return loss_value, accuracy

### Compute accuracy
def compute_accuracy_2d(pred, labels, ac_threshold):

	n_pixels = pred.shape[0]*pred.shape[1]*pred.shape[2]
	accuracy = torch.sum( torch.abs( (labels-pred)/(labels+config.eps_stability) ).le(ac_threshold) ).item()
	# accuracy = torch.sum( torch.abs( (labels-pred)).le(ac_threshold) ).item()
	accuracy /= n_pixels
	return accuracy

### Forward by batch
def forward_by_batch(data, model, batch_size):

	# Only for forward computation
	with torch.no_grad():

		# Allocate output prediction array on device
		y_pred = torch.zeros((data.shape[0], data.shape[1], data.shape[2]), device=config.device)
		# Compute total number of batches
		if batch_size == None:
			n_batch = 1
			batch_size = data.shape[0]
		else:
			n_batch_full = data.shape[0]//batch_size
			n_batch = n_batch_full
			if data.shape[0]%batch_size != 0:
				n_batch = n_batch_full+1

		# Loop over mini-batches
		for i_batch in range(n_batch):
			# Get min/max index
			idx_min = i_batch*batch_size
			idx_max = min( (i_batch+1)*batch_size, data.shape[0] )
			# Compute forward pass for this batch
			y_pred[idx_min:idx_max,:,:] = model.forward(data[idx_min:idx_max,:,:,:])

	return y_pred

################################################################################
################################## Training ####################################
################################################################################
## Training main loop
def fit_model_2d(name, all_data, all_labels, model, info=True):

	# Create pointers to data/labels
	data_train = all_data['train']
	labels_train = all_labels['train']
	data_dev = all_data['dev']
	labels_dev = all_labels['dev']

	# Get other parameters from config
	num_epochs = config.num_epochs
	learning_rate = config.learning_rate
	l2_reg_lambda = config.l2_reg_lambda
	batch_size = config.batch_size
	optim_method = config.optim_method
	optimizer = get_optim(optim_method, model, learning_rate, l2_reg_lambda)
	if config.half == True:
		half = True
		scaler = torch.cuda.amp.GradScaler()
		print("Using half precision")
	else:
		half = False

	# Learning rate decay scheduler
	if config.lr_decay_strategy != None:
		lr_decay = True
		lr_scheduler = create_scheduler(optimizer, num_epochs)
	else:
		lr_decay = False

	# Display optimization parameters
	if info:
		print("-"*30)
		print("Model name: ", name)
		print("Data type: ", type(all_data))
		print("Labels type: ", type(all_labels))
		print("Model: ", model)
		print("Optimization method: ", optim_method)
		print("Number of epochs: ", num_epochs)
		print("Using a learning rate decay strategy")
		print("Batch size: ", batch_size)
		print("-"*30)

	### Loss function
	loss_function = get_loss_fn(config.loss_function)

	### Inversion metrics
	loss_train = []
	loss_dev = []
	accuracy_train = []
	accuracy_dev = []
	lr_curve =[]

	# Compute initial train loss
	loss_value, accuracy = compute_loss_2d(data_train, labels_train, model, loss_function)
	if 'cuda' in config.device: loss_value = loss_value.cpu()
	loss_train.append(loss_value)
	accuracy_train.append(accuracy)

	# Compute initial dev loss
	loss_value_dev, dev_accuracy = compute_loss_2d(data_dev, labels_dev, model, loss_function)
	if 'cuda' in config.device: loss_value_dev = loss_value_dev.cpu()
	loss_dev.append(loss_value_dev)
	accuracy_dev.append(dev_accuracy)

	# Learning rate scheduler
	lr_curve.append(optimizer.param_groups[0]['lr'])

	# Display information
	if info:
		print("Initial learning rate: ", optimizer.param_groups[0]['lr'])
		print("Epoch #", 0, "/", num_epochs)
		print("Initial train loss: ", loss_value.item())
		print("Initial train accuracy: ", accuracy)

	### Train model for num_epoch
	for i_epoch in range(num_epochs):

		########################################################################
		############### Update model parameters for one epoch ##################
		########################################################################

		# update_model_epoch(data_train, labels_train, model, batch_size, optimizer, loss_function, info)
		# Compute total number of batches
		if batch_size == None:
			n_batch = 1
			batch_size = data_train.shape[0]
		else:
			n_batch_full = data_train.shape[0]//batch_size
			n_batch = n_batch_full
			if data_train.shape[0]%batch_size != 0:
				n_batch = n_batch_full+1

		# Create permutation for data shuffling
		permutation = torch.randperm(data_train.shape[0])

		# If data_train size not divisible by batch size
		disp=False
		if disp:
			if data_train.shape[0]%batch_size != 0:
				print("Warning: size of training data (", data_train.shape[0], ") is not divisible by batch size (", batch_size, ")")
				print("Number of batches of full size: ", n_batch-1)
				print("Size of last batch: ", data_train.shape[0]-n_batch_full*batch_size)
			else:
				print("Size of training data (", data_train.shape[0], ") is divisible by batch size (", batch_size, ")")
				print("Number of batches of full size: ", n_batch)

		# Loop over mini-batches
		for i_batch in range(n_batch):

			# Get indices for permuation array
			idx_min = i_batch*batch_size
			idx_max = min( (i_batch+1)*batch_size, data_train.shape[0] )
			indices = permutation[idx_min:idx_max]

			# Extract the batch from data and label
			data_train_batch = data_train[indices,:,:,:]
			labels_train_batch = labels_train[indices,:,:]

			########################### Half precision ###########################
			if half:

				with amp.autocast():

					# Compute forward pass
					y_pred_batch = model.forward(data_train_batch)
					# Compute loss
					loss = loss_function(y_pred_batch, labels_train_batch)
					CTP_utils_2d.get_cuda_info_2d("Memory status for this epoch - here 4")

				# Set gradients to zero
				optimizer.zero_grad()

				# Compute gradient
				scaler.scale(loss).backward()

				# Update model parameters
				scaler.step(optimizer)
				scaler.update()

			############################# Float ################################
			else:

				# Set gradients to zero
				optimizer.zero_grad()

				# Forward pass
				y_pred_batch = model.forward(data_train_batch)

				# Compute loss
				loss = loss_function(y_pred_batch, labels_train_batch)

				# Compute gradient
				loss.backward()

				# Update model parameters
				optimizer.step()

		########################################################################
		# Compute loss function and accuracy on the full train set
		# Half precision
		if half:
			with amp.autocast():
				# Training set
				loss_value, accuracy = compute_loss_2d(data_train, labels_train, model, loss_function)
				if 'cuda' in config.device: loss_value = loss_value.cpu()
				loss_train.append(loss_value)
				accuracy_train.append(accuracy)

				# Dev set
				loss_value_dev, dev_accuracy = compute_loss_2d(data_dev, labels_dev, model, loss_function)
				if 'cuda' in config.device: loss_value_dev = loss_value_dev.cpu()
				loss_dev.append(loss_value_dev)
				accuracy_dev.append(dev_accuracy)


		# Float32
		else:
			# Training set
			loss_value, accuracy = compute_loss_2d(data_train, labels_train, model, loss_function)
			if 'cuda' in config.device: loss_value = loss_value.cpu()
			loss_train.append(loss_value)
			accuracy_train.append(accuracy)

			# Dev set
			loss_value_dev, dev_accuracy = compute_loss_2d(data_dev, labels_dev, model, loss_function)
			if 'cuda' in config.device: loss_value_dev = loss_value_dev.cpu()
			loss_dev.append(loss_value_dev)
			accuracy_dev.append(dev_accuracy)

		if (i_epoch+1)%config.rec_freq == 0:
			print("-"*40)
			print("Epoch #", i_epoch+1, "/", num_epochs)
			print("Train loss: ", loss_value.item())
			print("Train loss decreased by ", np.absolute(loss_value.item()-loss_train[0])/loss_train[0], " %")
			print("Train accuracy: ", accuracy)
			print("Dev loss: ", loss_value_dev)
			print("Dev loss decreased by ", np.absolute(loss_value_dev.item()-loss_dev[0])/loss_dev[0], " %")
			print("Dev accuracy: ", dev_accuracy)
			print("Learning rate value: ", optimizer.param_groups[0]['lr'])
			CTP_utils_2d.get_cuda_info_2d("GPU info for epoch #"+str(i_epoch+1))

		# Update learning rate
		if lr_decay:
			lr_scheduler.step()
		lr_curve.append(optimizer.param_groups[0]['lr'])

	return model, loss_train, loss_dev, accuracy_train, accuracy_dev, lr_curve
