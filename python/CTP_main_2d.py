import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import copy
import argparse
import os, sys
from os import path
import time
from CTP_config import config
import CTP_utils_2d
import CTP_models_2d
import CTP_train_2d

################################################################################
################################## Training ####################################
################################################################################
def train_model_2d(project_folder, project_name):
	
	# Display information
	info = config.info
	debug = config.debug
	start_time = time.time()

	if info:
		print("[train_model_2d] User has requested to display information")
		print("[train_model_2d] Project name: ", project_name)
		print("[train_model_2d] Project folder: ", project_folder)
		print("[train_model_2d] Training mode")

	###################### Load data + pre-processing ##########################
	# Load datasets
	data_train, labels_train, geometry_train = CTP_utils_2d.load_data_2d("train")

	# Dev set
	data_dev, labels_dev, geometry_dev = CTP_utils_2d.load_data_2d("dev")

	print("[train_model_2d] data_train shape: ", data_train.shape)
	print("[train_model_2d] labels_train shape: ", labels_train.shape)
	print("[train_model_2d] data_dev shape: ", data_dev.shape)
	print("[train_model_2d] labels_dev shape: ", labels_dev.shape)
	print("[train_model_2d] geometry_train shape: ", geometry_train.shape)
	print("[train_model_2d] geometry_dev shape: ", geometry_dev.shape)

	n_train = data_train.shape[0]
	ny = data_train.shape[1]
	nx = data_train.shape[2]
	nt = data_train.shape[3]

	if info:
		print("[train_model_2d] Number of training examples: ", n_train)
		print("[train_model_2d] ny: ", ny)
		print("[train_model_2d] nx: ", nx)
		print("[train_model_2d] nt: ", nt)

	# Normalize data
	train_stats = {}
	train_stats['mean'] = np.mean(data_train)
	train_stats['std'] = np.std(data_train)
	train_stats['n_train'] = n_train
	data_train = (data_train - train_stats['mean']) / train_stats['std']
	data_dev = (data_dev - train_stats['mean']) / train_stats['std']

	############################ Data statistics ###############################
	if info:
		print("-"*10, "Data statistics", "-"*10)
		print("[train_model_2d] Mean training examples: ", train_stats['mean'])
		print("[train_model_2d] Standard deviation of training examples: ", train_stats['std'])
		print("[train_model_2d] Min value train data: ", np.min(data_train))
		print("[train_model_2d] Max value train data: ", np.max(data_train))
		print("[train_model_2d] Min value train labels: ", np.min(labels_train))
		print("[train_model_2d] Max value train labels: ", np.max(labels_train))
		print("-"*40)

	############################# Torch ########################################
	### Convert data and labels into PyTorch tensors
	data_train_torch = CTP_utils_2d.np2torch_2d(data_train, config.device)
	data_dev_torch = CTP_utils_2d.np2torch_2d(data_dev, config.device)
	labels_train_torch = CTP_utils_2d.np2torch_2d(labels_train, config.device)
	labels_dev_torch = CTP_utils_2d.np2torch_2d(labels_dev, config.device)
	all_data = {}
	all_labels = {}
	all_data['train'] = data_train_torch
	all_labels['train'] = labels_train_torch
	all_data['dev'] = data_dev_torch
	all_labels['dev'] = labels_dev_torch
	# CTP_utils_2d.get_cuda_info_2d("[train_model_2d] After loading data on GPU")
	########################### Model instanciation ############################
	# Create model
	model = CTP_models_2d.create_model_2d(config.model_type, nt, config.device)

	# Initialize weights
	model.initialize_weights()

	# Log type of model in train statistics dictionary
	train_stats['model_type'] = model.name
	train_stats['model_n_param'] = sum(p.numel() for p in model.parameters())

	################################ Training ##################################
	# Train model
	model_out, loss_train, loss_dev, accuracy_train, accuracy_dev, lr_curve = CTP_train_2d.fit_model_2d(model.name, all_data, all_labels, model, info=info)

	# Record training information
	training_time = time.time()-start_time
	train_stats['training time'] = training_time
	train_stats['device'] = config.device
	train_stats['optim_method'] = config.optim_method
	train_stats['batch_size'] = config.batch_size
	train_stats['lr_decay_strategy'] = config.lr_decay_strategy
	train_stats['loss_function'] = config.loss_function
	train_stats['seed'] = config.seed
	print("Training time: %2.1f [s]" %(training_time))

	############################ Saving + plotting #############################
	# Save model
	if config.save_model:
		if info: print("Saving model")
		CTP_utils_2d.save_model_2d(model_out, train_stats, project_folder, project_name)

	# Plot and save train/dev loss functions
	if config.save_loss:
		if info: print("Saving loss and accuracy")
		CTP_utils_2d.save_results_2d(loss_train, loss_dev, accuracy_train, accuracy_dev, lr_curve, project_folder, project_name)

	# Compute and save prediction
	if config.save_pred:
		if info: print("Saving prediction")
		model_out.eval()
		with torch.no_grad():
			y_pred_train = CTP_train_2d.forward_by_batch(all_data['train'], model_out, config.batch_size)
			y_pred_dev = CTP_train_2d.forward_by_batch(all_data['dev'], model_out, config.batch_size)
		print("y_pred_train shape: ", y_pred_train.shape)
		print("y_pred_dev shape: ", y_pred_dev.shape)
		print("Saving predictions")
		CTP_utils_2d.save_pred_2d(y_pred_train, all_labels['train'], geometry_train, y_pred_dev, all_labels['dev'], geometry_dev, project_folder, project_name)

if __name__ == '__main__':

	############################################################################
	############################### Main #######################################
	############################################################################
	# Parse command line
	parser = argparse.ArgumentParser(description=' ')
	parser.add_argument("mode", help="Select between train/test/predict modes", type=str)
	parser.add_argument("project_name", help="Name of the experiment", type=str)
	parser.add_argument("--patient_file", help="Patient's file (h5 format)", type=str)
	parser.add_argument("--patient_id", help="Patient's id", type=str)
	parser.add_argument("--n_epochs", help="Number of epochs for training", type=int)
	parser.add_argument("--optim", help="Optimization method", type=str)
	parser.add_argument("--lr", help="Learning rate", type=float)
	parser.add_argument("--model", help="Model type", type=str)
	parser.add_argument("--at", help="Accuracy threashold (in tenth of a sec)", type=float)
	parser.add_argument("--batch_size", help="Batch size (set to -1 for batch gradient)", type=int)
	parser.add_argument('--lr_decay', type=str)
	parser.add_argument('--device', help="Device for computation ('cpu' or 'cuda')", type=str)
	parser.add_argument('--train_file', help="Name of file for training", type=str)
	parser.add_argument('--train_file_list', help="Name of file for training", type=str)
	parser.add_argument('--dev_file', help="Name of file for dev", type=str)
	parser.add_argument('--dev_file_list', help="Name of file for dev", type=str)
	parser.add_argument('--test_file', help="Name of file for test", type=str)
	parser.add_argument('--test_file_list', help="Name of file for test", type=str)
	parser.add_argument('--decay_rate', help="Decay rate for the decay lr schedule", type=float)
	parser.add_argument('--decay_gamma', help="Decay rate for the exp decay lr schedule", type=float)
	parser.add_argument('--step_size', help="Learning rate update frequency (for step schedule)", type=float)
	parser.add_argument('--loss', help="Loss function", type=str)
	parser.add_argument('--l2_reg_lambda', help="Trade-off parameter for L2-regularization", type=float)
	parser.add_argument('--seed', help="Use a seed to obtain deterministic results", type=int)
	parser.add_argument('--half', help="Set to 1 to use half (mixed) precision on GPU", type=int)
	parser.add_argument('--batch_size_time', help="Batch size for time encoder", default=512, type=int)
	args = parser.parse_args()
	curr_dir = os.getcwd()
	project_folder = curr_dir + '/models_2d/' + args.project_name

	# Override config parameters
	if args.n_epochs != None: config.num_epochs = args.n_epochs
	if args.optim != None: config.optim_method = args.optim
	if args.lr != None: config.learning_rate = args.lr
	if args.model != None: config.model_type = args.model
	if args.at != None: config.ac_threshold = args.at
	if args.batch_size != None:
		config.batch_size = None if args.batch_size == -1 else args.batch_size
	if args.at != None: config.ac_threshold = args.at
	config.lr_decay_strategy = args.lr_decay
	if args.device != None: config.device = args.device
	if args.decay_rate != None: config.decay_rate = args.decay_rate
	if args.decay_gamma != None: config.decay_gamma = args.decay_gamma
	if args.step_size != None: config.decay_step_size = args.step_size
	if args.loss != None: config.loss_function = args.loss
	if args.l2_reg_lambda != None: config.l2_reg_lambda = args.l2_reg_lambda
	if isinstance(args.seed, int):
		config.seed = args.seed
		CTP_utils_2d.seed_everything(config.seed)
	if isinstance(args.half, int):
		if args.half == 1:
			config.half=True
		else:
			config.half=False
	config.batch_size_time = args.batch_size_time

	# Training data information
	config.train_file_list = []
	if args.train_file != None:
		config.train_file_list.append(args.train_file)
	elif args.train_file_list != None:
		with open(args.train_file_list) as f:
			for row in f:
				config.train_file_list.append(row.rstrip('\n') )

	# Dev data information
	if args.dev_file != None:
		config.dev_file_list.append(args.dev_file)
	elif args.dev_file_list != None:
		with open(args.dev_file_list) as f:
			for row in f:
				config.dev_file_list.append(row.rstrip('\n') )

	# Test data information
	if args.test_file != None:
		config.test_file_list.append(args.test_file)
	elif args.test_file_list != None:
		with open(args.test_file_list) as f:
			for row in f:
				config.test_file_list.append(row.rstrip('\n') )

	# Display info
	if config.info:
		print("----Parameters for training----")
		print("Model type: ", config.model_type)
		print("Number of epochs: ", config.num_epochs)
		print("Optimization method: ", config.optim_method)
		print("Learning rate: ", config.learning_rate)
		print("Accuracy threshold: ", config.ac_threshold)
		print("Accuracy stability value: ", config.eps_stability)
		print("Learning rate decay strategy: ", config.lr_decay_strategy)
		print("Device: ", config.device)
		print("Train file_list: ", config.train_file_list)
		print("Dev file_list: ", config.dev_file_list)
		print("Test file_list: ", config.test_file_list)
		print("Batch size: ", config.batch_size)
		print("Batch size for the time encoder: ", config.batch_size_time)
		print("Loss function: ", config.loss_function)
		if config.half == 1: print("Using half precision")
		if config.seed != None: print("Seed for deterministic results: ", config.seed)

	# Train mode
	if args.mode == 'train':

		# Create working directory for this model
		if path.exists(project_folder): sys.exit('Path for experiment already exist, please choose another name')
		os.mkdir(project_folder)

		# Train model
		train_model_2d(project_folder, args.project_name)
