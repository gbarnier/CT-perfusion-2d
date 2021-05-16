#! /usr/bin/env python3
import argparse
import numpy as np
import h5py
from CT_utils import *
from matplotlib import pyplot as plt
import copy
import random
from CTP_config import config
import pyVector
import os
import sys

if __name__ == "__main__":

    ########################## Parsing command line ############################
    parser = argparse.ArgumentParser(description='Program to create patient formatted file for Tmax estimation')
    parser.add_argument("dcm_dir", help="Directory path containing the DCM files of a patient", type=str)
    parser.add_argument("tmax_dir", help="Directory containing the Tmax value file; if multiple directories, use colon-separated paths", type=str)
    parser.add_argument("wind_cent", help="Windowing center value", type=float)
    parser.add_argument("wind_widt", help="Windowing width value", type=float)
    parser.add_argument("out_file", help="Output HDF5 file contaning processed data", type=str)
    parser.add_argument("--verbose","-v", help="Verbosity level of the code", default=True, type=bool)
    parser.add_argument("--size","-s", help="Spatial average filter width", default=2, type=int)
    parser.add_argument("--desample","-d", help="Spatial desampling", default=2, type=int)
    parser.add_argument("--raw", help="Provide raw CT data (1/0)", type=int)
    parser.add_argument("--SEP", help="Output SEP files", default=1, type=int)
    parser.add_argument("--x_patch_size", help="Patch size in the x-direction", default=256, type=int)
    parser.add_argument("--y_patch_size", help="Patch size in the y-direction", default=256, type=int)
    parser.add_argument("--ox_patch", help="Coordinate of origin in the x-direction", default=0.0, type=float)
    parser.add_argument("--oy_patch", help="Coordinate of origin in the x-direction", default=0.0, type=float)
    parser.add_argument("--nx_patch", help="Number of patches the x-direction", default=1, type=int)
    parser.add_argument("--ny_patch", help="Number of patches in the y-direction", default=1, type=int)
    parser.add_argument("--skip", nargs='+', help="List of axial slices to skip", default=-1, type=int)
    parser.add_argument("--halo", help="Provide length of halo (in samples)", default=0, type=int)
    parser.add_argument("--clip_tmax", help="Provide length of halo (in samples)", default=-1.0, type=float)
    args = parser.parse_args()

    ############################################################################
    ############################### Parameters #################################
    ############################################################################
    verb = args.verbose
    wind_c = args.wind_cent # Hounsfield units window center
    wind_w = args.wind_widt # Hounsfield units window width
    out_file = args.out_file
    size = args.size # Spatial average filter width
    desample = args.desample # Spatial average filter width
    raw = True if args.raw == 1 else False
    SEP = True if args.SEP == 1 else False
    x_patch_size = args.x_patch_size
    y_patch_size = args.y_patch_size
    ox_patch = args.ox_patch
    oy_patch = args.oy_patch
    nx_patch = args.nx_patch
    ny_patch = args.ny_patch
    halo = args.halo
    skip = args.skip
    clip_tmax = args.clip_tmax
    print("skip: ", skip)
    print("skip type: ", type(skip))
    print("Window center [HU]: ", wind_c)
    print("Window width [HU]: ", wind_w)
    print("Output file: ", out_file)
    print("Spatial average filter width: ", args.size)
    print("clip_tmax: ", clip_tmax)
    if raw: print("User has requested to output raw data")
    if SEP: print("User has requested to output SEP files")

    # Getting file folders
    dmc_dir = args.dcm_dir
    tmax_dir = args.tmax_dir

    # Adding final slash if necessary
    dmc_dirs = dmc_dir.split(":")
    for idx in range(len(dmc_dirs)):
        if dmc_dirs[idx][-1] != "/":
            dmc_dirs[idx] += "/"
    tmax_dirs = tmax_dir.split(":")
    for idx in range(len(tmax_dirs)):
        if tmax_dirs[idx][-1] != "/":
            tmax_dirs[idx] += "/"


    print("DCM directories: ", dmc_dirs)
    print("tmax directories: ", tmax_dirs)

    # Processing slices
    ct_image, x_axis_ct, y_axis_ct, z_axis_ct, time_axes = read_slices(dmc_dirs, verbose=verb)

    # Rotation
    ct_image, ang = image4D_rotation(ct_image)

    # Co-registration
    ct_image_reg = image4D_registration(ct_image)

    # Skull-stripping
    inner_mask, use_Tmax = skull_strip_mask_4d(ct_image_reg)
    if isinstance(skip, list):
        for i_slice in range(len(skip)):
            use_Tmax[skip[i_slice]] = 0

        # Windowing
    ct_image_reg_wind = window_image4D(ct_image_reg, wind_c, wind_w)

    # Space averaging
    ct_image_reg_wind = spatial_ave_image4d(ct_image_reg_wind, size=size)[:,:,::desample,::desample]
    x_axis_ct = x_axis_ct[::desample]
    y_axis_ct = y_axis_ct[::desample]
    inner_mask = inner_mask[:,::desample,::desample]
    use_Tmax_reshape = np.reshape(use_Tmax,(use_Tmax.shape[0],1,1))
    skull_mask = inner_mask*use_Tmax_reshape

    # QC
    print("inner mask shape: ", inner_mask.shape)
    print("use_Tmax shape: ", use_Tmax.shape)
    print("use_Tmax reshape shape: ", use_Tmax_reshape.shape)
    print("skull_mask shape: ", skull_mask.shape)
    print("use_Tmax: ", use_Tmax)

    # Compute spatial sampling
    dx = x_axis_ct[1]-x_axis_ct[0]
    dy = y_axis_ct[1]-y_axis_ct[0]
    dz = z_axis_ct[1]-z_axis_ct[0]

    # Get axis information
    ox = x_axis_ct[0]
    oy = y_axis_ct[0]
    oz = z_axis_ct[0]

    # Find min index
    ix_min = int((ox_patch-ox)/dx)
    iy_min = int((oy_patch-oy)/dy)

    print("Minimum x-position patch: ", ox_patch)
    print("Minimum index patch x-axis: ", ix_min)
    print("Minimum y-position patch: ", oy_patch)
    print("Minimum index patch y-axis: ", iy_min)

    # Halo
    x_patch_size_halo = x_patch_size + 2*halo
    y_patch_size_halo = y_patch_size + 2*halo
    print("x-patch size: ", x_patch_size)
    print("y-patch size: ", y_patch_size)
    print("x-patch size with halo: ", x_patch_size_halo)
    print("y-patch size with halo: ", y_patch_size_halo)
    # if config.x_patch_size != x_patch_size_halo or config.y_patch_size != y_patch_size_halo: sys.exit('Patch size + halo different than patch size in configuration file')
    ix_min_halo = ix_min - halo
    iy_min_halo = iy_min - halo
    print("Minimum index patch y-axis with halo: ", iy_min_halo)
    print("Minimum index patch x-axis with halo: ", ix_min_halo)
    print("x-patch size: ", x_patch_size)
    print("y-patch size: ", y_patch_size)
    print("x-patch size with halo: ", x_patch_size_halo)
    print("y-patch size with halo: ", y_patch_size_halo)

    # QC time axis
    large_gap = False # Flag to check if a larger time sampling is deteched
    if np.diff(time_axes).max() > 4.0:
        large_gap = True
        print("WARNING! Large time sampling gap detected for %s" % dmc_dir)

    # Interpolate time axis
    ct_image_reg_wind, time_axis = interpolate_time_image4d(ct_image_reg_wind, time_axes)
    n_rect = 6

    # Filtering in time (high-cut filter)
    ct_image_reg_wind = filter_time_image4d(ct_image_reg_wind, n_rect)
    data_train_temp = ct_image_reg_wind*skull_mask

    ############################# Labels processing ############################
    # Reading Tmax file
    tmax, tmax_z_axis = read_Tmax(tmax_dirs)
    tmax_raw = copy.deepcopy(tmax)
    tmax_z_axis_raw = copy.deepcopy(tmax_z_axis)
    tmax = tmax_rotation(tmax, ang)
    min_tmax = np.amin(tmax)
    if clip_tmax > 0:
        print("Applying clipping to TMax at: ", clip_tmax)
        print("Min before", np.amin(tmax))
        print("Max before", np.amax(tmax))
        tmax = np.clip(tmax, min_tmax, clip_tmax)
        print("Min after", np.amin(tmax))
        print("Max after", np.amax(tmax))
    labels_train_temp = tmax*skull_mask

    ############################# QC values ####################################
    # QC
    print("data_train_temp shape: ", data_train_temp.shape)
    print("labels_train_temp shape: ", labels_train_temp.shape)
    print("Average data values: ", np.mean(data_train_temp))
    print("Min data values: ", np.min(data_train_temp))
    print("Max data values: ", np.max(data_train_temp))
    print("Average labels values: ", np.mean(labels_train_temp))
    print("Min labels values: ", np.min(labels_train_temp))
    print("Max labels values: ", np.max(labels_train_temp))

    ########################## Slice data / labels #############################
    # Compute patch size
    nt = data_train_temp.shape[0]
    nz = data_train_temp.shape[1]
    ny = data_train_temp.shape[2]
    nx = data_train_temp.shape[2]
    # if ny%y_patch_size != 0: sys.exit('ny not divisible by y_patch_size')
    # ny_patch = int(ny/y_patch_size)
    # if nx%x_patch_size != 0: sys.exit('nx not divisible by x_patch_size')
    # nx_patch = int(nx/x_patch_size)
    # n_patch_per_slice = int(ny_patch*nx_patch)
    # ny_patch = 3
    # nx_patch = 3
    n_patch_per_slice = int(ny_patch*nx_patch)
    print("ny_patch: ", ny_patch)
    print("nx_patch: ", nx_patch)
    print("n_patch_per_slice: ", n_patch_per_slice)
    # Number of axial slices kept
    nb_axial_slice = int(np.sum(use_Tmax))

    # Geometry
    geometry = np.zeros((1, 4))
    geometry[0,0] = nb_axial_slice
    geometry[0,1] = int(nx_patch)
    geometry[0,2] = int(ny_patch)
    geometry[0,3] = int(halo)
    print("Geometry: ", geometry)

    # Allocate data array
    n_train = int(nb_axial_slice*n_patch_per_slice)
    print("n_train: ", n_train)
    print("ny_patch: ", ny_patch)
    print("nx_patch: ", nx_patch)
    data_train = np.zeros((n_train, y_patch_size_halo, x_patch_size_halo, nt))
    # data_train_m = np.zeros((n_train, y_patch_size, x_patch_size, nt))
    labels_train = np.zeros((n_train, y_patch_size_halo, x_patch_size_halo))
    # labels_train_halo = np.zeros((n_train, y_patch_size, x_patch_size))

    print("Number of slices used: ", nb_axial_slice)
    print("Number of examples to train on: ", n_train)

    i_train=0
    for iz in range(nz):
        if use_Tmax[iz] != 0:
            # Loop over patches on y-direction
            for iy in range(ny_patch):
                idy_min = iy_min+iy*y_patch_size-halo # y-index on large grid
                # Loop over patches on x-direction
                for ix in range(nx_patch):
                    idx_min = ix_min+ix*x_patch_size-halo # x-index on large grid
                    # loop within a patch
                    for iy_small in range(y_patch_size_halo):
                        for ix_small in range(x_patch_size_halo):

                            labels_train[i_train,iy_small,ix_small] = labels_train_temp[iz,idy_min+iy_small,idx_min+ix_small]
                            max_val = np.max(data_train_temp[:,iz,idy_min+iy_small,idx_min+ix_small])
                            if max_val>200 and labels_train_temp[iz,idy_min+iy_small,idx_min+ix_small]==0:
                                for it in range(nt):
                                    data_train[i_train,iy_small,ix_small,it] = 0
                                    # data_train[i_train,iy_small,ix_small,it] = data_train_temp[it,iz,idy_min+iy_small,idx_min+ix_small]
                            else:
                                for it in range(nt):
                                    data_train[i_train,iy_small,ix_small,it] = data_train_temp[it,iz,idy_min+iy_small,idx_min+ix_small]
                                    # data_train[i_train,iy_small,ix_small,it] = data_train_temp[it,iz,idy_min+iy_small,idx_min+ix_small]

                    i_train+=1

    ############################################################################
    ############################# Saving files #################################
    ############################################################################

    ################################# Qc #######################################
    if verb:
        print("Shape train data: ", data_train.shape)
        print("Shape train labels: ", labels_train.shape)
        print("Large gap?: ", large_gap)
        print("nx: ", x_axis_ct.shape[0], "dx: ", x_axis_ct[1]-x_axis_ct[0], "ox: ", x_axis_ct[0])
        print("ny: ", y_axis_ct.shape[0], "dy: ", y_axis_ct[1]-y_axis_ct[0], "oy: ", y_axis_ct[0])
        print("nz: ", z_axis_ct.shape[0], "dz: ", z_axis_ct[1]-z_axis_ct[0], "oz: ", z_axis_ct[0])
        print("nt: ", time_axis.shape[0], "dt: ", time_axis[1]-time_axis[0], "ot: ", time_axis[0])
        if raw:
            print("Shape 4D raw data: ", ct_image_raw.shape)
            print("Shape 3D raw TMax: ", tmax_raw.shape)
            print("raw nx: ", x_axis_ct_raw.shape[0], "raw dx: ", x_axis_ct_raw[1]-x_axis_ct_raw[0], "raw ox: ", x_axis_ct_raw[0])
            print("raw ny: ", y_axis_ct_raw.shape[0], "raw dy: ", y_axis_ct_raw[1]-y_axis_ct_raw[0], "raw oy: ", y_axis_ct_raw[0])
            print("raw nz: ", z_axis_ct_raw.shape[0], "raw dz: ", z_axis_ct_raw[1]-z_axis_ct_raw[0], "raw oz: ", z_axis_ct_raw[0])
            print("raw nt: ", time_axis_raw.shape[0], "raw dt: ", time_axis_raw[1]-time_axis_raw[0], "raw ot: ", time_axis_raw[0])

    ############################### Processed data #############################
    # HF files
    if verb:
        print("Writing output .h5 file: ", out_file)
    hf = h5py.File(out_file, 'w')
    hf.create_dataset('data_train', data=data_train)
    # hf.create_dataset('data_train_m', data=data_train_m)
    hf.create_dataset('labels_train', data=labels_train)
    hf.create_dataset('tmax', data=tmax_raw)
    hf.create_dataset('geometry', data=geometry)
    hf.close()

    # Write SEP files
    if SEP:
        if verb:
            print("Writing file for processed data in SEP format")
        hf = h5py.File(out_file, 'r')

        # Training data
        data_train = np.array(hf.get("data_train"), dtype=np.float32)
        vec = pyVector.vectorIC(data_train)
        vec.writeVec(out_file+"_data_train.H")
        command="echo 'o2="+str(ox)+" d2="+str(dx)+" o3="+str(oy)+" d3="+str(dy)+"'>> "+out_file+"_data_train.H"
        os.system(command)

        # Training data with mask
        geometry = np.array(hf.get("geometry"), dtype=np.float32)
        vec = pyVector.vectorIC(geometry)
        vec.writeVec(out_file+"_geometry.H")

        # Training lables
        labels_train = np.array(hf.get("labels_train"), dtype=np.float32)
        vec = pyVector.vectorIC(labels_train)
        vec.writeVec(out_file+"_labels_train.H")
        command="echo 'o1="+str(ox)+" d1="+str(dx)+" o2="+str(oy)+" d2="+str(dy)+"'>> "+out_file+"_labels_train.H"
        os.system(command)

        # Training lables
        tmax_original = np.array(hf.get("tmax"), dtype=np.float32)
        vec = pyVector.vectorIC(tmax_original)
        vec.writeVec(out_file+"_tmax.H")
        os.system(command)
