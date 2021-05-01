from CTP_config import config
import numpy as np
import torch
import h5py
# import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import ast
from numpy import savetxt
import pyVector
import os
from CTP_config import config
from scipy.ndimage import uniform_filter
import random

################################################################################
#################################### GPU #######################################
################################################################################
# GPU transfer
def np2torch_2d(np_array, device, cast_double_to_float=True):

	"""
	Utility function that accepts a numpy array and does the following:
		1. Convert to torch tensor
		2. Move it to the GPU (if CUDA is available)
		3. Optionally casts float64 to float32 (torch is picky about types)
	"""

	# Move to CPU/GPU if requested
	np_array = torch.from_numpy(np_array).to(device)

	if cast_double_to_float and np_array.dtype is torch.float64:
		np_array = np_array.float()
	return np_array

# Memory info
def get_cuda_info_2d(msg=None):
	print("-"*20, "GPU info", "-"*20)
	if msg != None: print(msg)
	if torch.cuda.is_available(): print("A GPU device is available for this session")
	print("Device requested by user: ", config.device)
	print("Current device id: ", torch.cuda.current_device())
	print("Current device name: ", torch.cuda.get_device_name())
	print("Number of available GPUs: ", torch.cuda.device_count())
	print("Allocated memory on GPU: %3.2f [GB]" % (torch.cuda.memory_allocated(device=config.device)/1024**3))
	print("Reserved memory on GPU: %3.2f [GB]" % (torch.cuda.memory_reserved(device=config.device)/1024**3))
	print("Max allocated memory on GPU: %3.2f [GB]" % (torch.cuda.max_memory_allocated(device=config.device)/1024**3))
	print("Max reserved memory on GPU: %3.2f [GB]" % (torch.cuda.max_memory_reserved(device=config.device)/1024**3))
	print("-"*41+"-"*len("GPU info"))

# Seed
def seed_everything(seed=42):
	os.environ['PYTHONHASHSEED'] = str(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

################################################################################
################################# Data loading #################################
################################################################################
# Function loads train and dev data
def load_data_2d(dataset, debug=True):

	if dataset == "train":
		file_list = config.train_file_list
	elif dataset=="dev":
		file_list = config.dev_file_list

	# Read h5 file
	hf = h5py.File(file_list[0], 'r')
	data = np.array(hf.get("data_train"), dtype=np.float32)
	n_data = data.shape[0]
	labels = np.array(hf.get('labels_train'), dtype=np.float32)
	geometry = np.array(hf.get('geometry'), dtype=np.float32)

	# QC
	print("[load_data_2d] Length of train file list: ", len(file_list))
	print("[load_data_2d] Batch size train: ", data.shape[0])
	print("[load_data_2d] Batch size label: ", labels.shape[0])

	# Case where we have more than one file to train on
	if len(file_list) > 1:

		for i_file in range(len(file_list)-1):

			# Read h5 file corresponding to train file list
			hf = h5py.File(file_list[i_file+1], 'r')

			# Get numpy array for training data
			data_temp = np.array(hf.get("data_train"), dtype=np.float32)
			data = np.concatenate((data, data_temp), axis=0)

			# Same stuff for labels
			labels_temp = np.array(hf.get("labels_train"), dtype=np.float32)
			labels = np.concatenate((labels, labels_temp), axis=0)

			# Get geometry info
			geometry_temp = np.array(hf.get('geometry'), dtype=np.float32)
			geometry = np.concatenate((geometry, geometry_temp), axis=0)

			# QC
			print("[load_data_2d] File loaded ", file_list[i_file+1])

	# QC
	print("[load_data_2d] data_train shape: ", data.shape)
	print("[load_data_2d] labels_train shape: ", labels.shape)
	print("[load_data_2d] geometry shape: ", geometry.shape)
	print("[load_data_2d] geometry: ", geometry)

	# Return data/labels
	return data, labels, geometry

# Save model parameters
def load_model_2d(exp_folder, exp_name, n_time):

	# Load dictionary
	load_path_stats = exp_folder +'/'+ exp_name + '_model_stats.csv'
	print("load_path_stats: ", load_path_stats)
	file = open(load_path_stats, "r")
	contents = file.read()
	model_stats = ast.literal_eval(contents)
	file.close()

	# Load model
	load_path_model = exp_folder + '/'+ exp_name + '_model.mod'
	print("load_path_model: ", load_path_model)
	model = CTP_models.create_model(config.model_type, n_time)
	model.load_state_dict(torch.load(load_path_model))
	model.eval()
	return model_stats, model

################################################################################
#################################### Results ###################################
################################################################################
# Save model parameters
def save_model_2d(model, train_stats, exp_folder, exp_name):

	# Save model parameters
	save_path_model = exp_folder + '/' + exp_name + '_model.mod'
	torch.save(model.state_dict(), save_path_model)

	# Save statistics
	save_path_stats = exp_folder + '/' + exp_name + '_model_stats.csv'
	f = open(save_path_stats,"w")
	f.write(str(train_stats))
	f.close()

# Save accuracy
def save_accuracy_2d(accuracy, exp_folder, exp_name):
	save_path_accuracy = exp_folder + '/' + exp_name + '_test_time_accuracy.csv'
	f = open(save_path_accuracy,"w")
	f.write(str(accuracy))
	f.close()

# Save loss
def save_loss_2d(loss, exp_folder, exp_name):
	save_path_loss = exp_folder + '/' + exp_name + '_test_time_loss.csv'
	f = open(save_path_loss,"w")
	f.write(str(loss))
	f.close()

# Plot and save objective functions for loss and accuracy
def save_results_2d(loss_train, loss_dev, accuracy_train, accuracy_dev, lr_curve, exp_folder, exp_name, show=False):

	# Convert list to numpy array
	loss_train = np.array(loss_train)
	accuracy_train = np.array(accuracy_train)
	lr_curve = np.array(lr_curve)
	if len(loss_dev) > 0:
		loss_dev = np.array(loss_dev)
		accuracy_dev = np.array(accuracy_dev)
		dev = True
	else: dev = False

	# Get paths
	save_path = exp_folder + '/' + exp_name

	# Save non-normalized train/dev loss functions values
	savetxt(save_path+'_loss_train.csv', loss_train, delimiter=',')
	savetxt(save_path+'_accuracy_train.csv', accuracy_train, delimiter=',')
	savetxt(save_path+'_lr_curve.csv', lr_curve, delimiter=',')
	if dev:
		savetxt(save_path+'_loss_dev.csv', loss_dev, delimiter=',')
		savetxt(save_path+'_accuracy_dev.csv', accuracy_dev, delimiter=',')

	save_plot=True
	if save_plot:

		# Create iteration axis
		num_epoch = loss_train.shape[0]
		epoch = np.arange(num_epoch)

		# Non-normalized objective functions
		f1 = plt.figure()
		plt.plot(epoch, loss_train, "b", label="Train loss", linewidth=config.linewidth)

		if dev: plt.plot(epoch, loss_dev, "r", label="Dev loss", linewidth=config.linewidth)
		plt.xlabel("Epochs")

		plt.ylabel(config.loss_function+" loss")
		plt.legend(loc="upper right")
		plt.title("Loss function ("+exp_name+")")

		max_val = np.max(np.maximum(loss_train, loss_dev))
		plt.ylim(0, max_val*1.05)
		plt.grid()

		if show: plt.show()
		f1.savefig(save_path + '_loss_fn.pdf', bbox_inches='tight')

		# Accuracy
		f1 = plt.figure()
		plt.plot(epoch, accuracy_train, "b", label="Accuracy train", linewidth=config.linewidth)
		if dev: plt.plot(epoch, accuracy_dev, "r", label="Accuracy dev", linewidth=config.linewidth)
		plt.xlabel("Epochs")
		plt.ylabel("Accuracy")
		plt.legend(loc="upper right")
		plt.title("Model accuracy ("+exp_name+")")
		plt.ylim(0,1.05)
		plt.grid()
		if show: plt.show()
		f1.savefig(save_path + '_accuracy_fn.pdf', bbox_inches='tight')

		# Learning rate curve
		f1 = plt.figure()
		# plt.plot(epoch, lr_curve, "-", label="Learning rate schedule")
		plt.plot(epoch, lr_curve, "-")
		plt.xlabel("Epochs")
		plt.ylabel("Learning rate")
		# plt.legend(loc="upper right")
		plt.title("Learning rate ("+exp_name+")")
		plt.ylim(0,lr_curve[0]*1.05)
		plt.grid()
		if show: plt.show()
		f1.savefig(save_path + '_lr_curve.pdf', bbox_inches='tight')

		f1 = plt.figure()
		lr_curve /= lr_curve[0]
		# plt.plot(epoch, lr_curve, "-", label="Normalized learning rate schedule")
		plt.plot(epoch, lr_curve, "-")
		plt.xlabel("Epochs")
		plt.ylabel("Normalized learning rate")
		# plt.legend(loc="upper right")
		plt.title("Normalized learning rate  ("+exp_name+")")
		plt.ylim(0,1.05)
		plt.grid()
		if show: plt.show()
		f1.savefig(save_path + '_lr_curve_norm.pdf', bbox_inches='tight')

		# Normalized objective functions
		loss_train /= loss_train[0]
		if dev: loss_dev /= loss_dev[0]
		f2 = plt.figure()
		plt.plot(epoch, loss_train, "b", label="Normalized train loss", linewidth=config.linewidth)
		if dev: plt.plot(epoch, loss_dev, "r", label="Normalized dev loss", linewidth=config.linewidth)
		plt.xlabel("Epochs")
		plt.ylabel("Normalized"+config.loss_function+" loss")
		plt.legend(loc="upper right")
		plt.title("Normalized loss function ("+exp_name+")")
		plt.ylim(0,1.05)
		plt.grid()
		if show: plt.show()
		f2.savefig(save_path + '_loss_fn_norm.pdf', bbox_inches='tight')

################################################################################
#################################### Prediction ################################
################################################################################
# Plot and save objective functions for loss and accuracy
def save_pred_2d(y_pred_train, labels_train, geometry_train, y_pred_dev, labels_dev, geometry_dev, project_folder, project_name):

	out_file = project_folder+'/'+project_name+'.h5'
	print("out_file: ", out_file)

	if 'cuda' in config.device:
		y_pred_train = y_pred_train.cpu()
		y_pred_dev = y_pred_dev.cpu()
		labels_train = labels_train.cpu().detach().numpy()
		labels_dev = labels_dev.cpu()

	print("max labels_train: ", np.amax(labels_train))
	print("max labels_train: ", np.amin(labels_train))

	write_pred_2d(y_pred_train, labels_train, geometry_train, out_file, "train")
	write_pred_2d(y_pred_dev, labels_dev, geometry_dev, out_file, "dev")

	# Save prediction in SEP format
	y_pred_train = np.float32(y_pred_train)
	y_pred_dev = np.float32(y_pred_dev)
	vec = pyVector.vectorIC(y_pred_train)
	vec.writeVec(out_file+"_y_pred_train.H")
	vec = pyVector.vectorIC(y_pred_dev)
	vec.writeVec(out_file+"_y_pred_dev.H")


def write_pred_2d(y_pred, labels, geometry, out_file, dataset):

	ids_small = 0
	for i_file in range(geometry.shape[0]):

		n_slice = int(geometry[i_file,0]) # Number of axial slices in training set
		nx_patch = int(geometry[i_file,1]) # Number of patchs in the x-direction per axial slice
		ny_patch = int(geometry[i_file,2]) # Number of patchs in the y-direction per axial slice
		halo = int(geometry[i_file,3]) # Length of halo on each side of the patch
		n_patch_slice = ny_patch*nx_patch # Number of patches per axial slice
		ny_halo = ny_patch*config.y_patch_size # Number of samples in the y-direction per axial slice
		nx_halo = nx_patch*config.x_patch_size # Number of samples in the x-direction per axial slice
		y_patch_size_no_halo = config.y_patch_size - 2*halo
		x_patch_size_no_halo = config.x_patch_size - 2*halo
		ny = ny_patch*(config.y_patch_size-2*halo) # Number of samples in the y-direction per axial slice without the halo
		nx = nx_patch*(config.x_patch_size-2*halo) # Number of samples in the x-direction per axial slice without the halo

		# Allocate arrays for full model
		y_pred_2d = np.zeros((n_slice, ny, nx), dtype=np.float32)
		labels_2d = np.zeros((n_slice, ny, nx), dtype=np.float32)
		y_pred_2d_halo = np.zeros((n_slice, ny_halo, nx_halo), dtype=np.float32)
		labels_2d_halo = np.zeros((n_slice, ny_halo, nx_halo), dtype=np.float32)

		for i_slice in range(n_slice):
			ids = int(i_slice/n_patch_slice)
			for iy in range(ny_patch):
				for ix in range(nx_patch):

					# Remove halos
					idy_min = iy*y_patch_size_no_halo
					idy_max = (iy+1)*y_patch_size_no_halo
					idx_min = ix*x_patch_size_no_halo
					idx_max = (ix+1)*x_patch_size_no_halo
					labels_2d[i_slice,idy_min:idy_max,idx_min:idx_max] = labels[ids_small,halo:halo+y_patch_size_no_halo,halo:halo+x_patch_size_no_halo]
					y_pred_2d[i_slice,idy_min:idy_max,idx_min:idx_max] = y_pred[ids_small,halo:halo+y_patch_size_no_halo,halo:halo+x_patch_size_no_halo]

					# Include halos
					idy_min = iy*config.y_patch_size
					idy_max = (iy+1)*config.y_patch_size
					idx_min = ix*config.x_patch_size
					idx_max = (ix+1)*config.x_patch_size
					labels_2d_halo[i_slice,idy_min:idy_max,idx_min:idx_max] = labels[ids_small,:,:]
					y_pred_2d_halo[i_slice,idy_min:idy_max,idx_min:idx_max] = y_pred[ids_small,:,:]
					ids_small += 1

		# Predictions on training set - 2D
		vec = pyVector.vectorIC(labels_2d)
		vec.writeVec(out_file+"_labels_2d_"+dataset+"_file"+str(i_file)+".H")
		vec = pyVector.vectorIC(y_pred_2d)
		vec.writeVec(out_file+"_y_pred_2d_"+dataset+"_file"+str(i_file)+".H")
		vec = pyVector.vectorIC(labels_2d_halo)
		vec.writeVec(out_file+"_labels_2d_halo_"+dataset+"_file"+str(i_file)+".H")
		vec = pyVector.vectorIC(y_pred_2d_halo)
		vec.writeVec(out_file+"_y_pred_2d_halo_"+dataset+"_file"+str(i_file)+".H")
