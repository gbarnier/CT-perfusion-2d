################################################################################
##################################### QC #######################################
################################################################################

##################################### 1 slice ##################################
# S00243 - 1 slice
dat_2d/data1-t1-64.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 64 --y_patch_size 64 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 30 --skip 0 1 2 4 5 6 7 8 9

dat_2d/data1-t1-192.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 192 --y_patch_size 192 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 30 --skip 0 1 2 4 5 6 7 8 9

# S00242 - 1 slice
dat_2d/data1-d1-64.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00242/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00242/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 64 --y_patch_size 64 --nx_patch 3 --ny_patch 3 --ox_patch 20 --oy_patch 20 --skip 0 1 2 3 5 6 7 8 9

dat_2d/data1-d1-192.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00242/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00242/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 192 --y_patch_size 192 --nx_patch 1 --ny_patch 1 --ox_patch 20 --oy_patch 20 --skip 0 1 2 3 5 6 7 8 9

makeData1:
	make dat_2d/data1-t1-64.h5 -B
	make dat_2d/data1-t1-192.h5 -B
	make dat_2d/data1-d1-64.h5 -B
	make dat_2d/data1-d1-192.h5 -B

# Training
train1:
	rm -rf models_2d/qc1
	python3.6 ./python/CTP_main_2d.py train qc1 --model EGNET --n_epochs 1000 --lr 5.0e-3 --device cuda:0 --train_file dat_2d/data1-t1-64.h5 --dev_file dat_2d/data1-d1-64.h5 --batch_size 64 --half 1 --seed 10

# Display
display1:
	Cp dat_2d/data1-t1-192.h5_labels_train.H t0.H
	Cp models_2d/qc1/qc1.h5_labels_train_2d_file0.H t1.H
	Cp models_2d/qc1/qc1.h5_y_pred_train_2d_file0.H t2.H
	Add t1.H t2.H scale=1,-1 > diff.H
	Attr < diff.H
	echo "d1=1.0 d2=1.0" >> t2.H
	Cat axis=3 t0.H t1.H t2.H | Grey color=j newclip=1 grid=y titles="Original:Labels:Pred" | Xtpen pximaps=y &
	Grey color=j newclip=1 < diff.H | Xtpen &

##################################### 1 head ###################################
# S00243 - 1 head
dat_2d/data2-t1-64.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 64 --y_patch_size 64 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 30

dat_2d/data2-t1-192.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 192 --y_patch_size 192 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 30

dat_2d/data2-t2-64.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 64 --y_patch_size 64 --nx_patch 2 --ny_patch 3 --ox_patch 40 --oy_patch 30

dat_2d/data2-t2-192.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 128 --y_patch_size 192 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 30

# S00242 - 1 head
dat_2d/data2-d1-64.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00242/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00242/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 64 --y_patch_size 64 --nx_patch 3 --ny_patch 3 --ox_patch 20 --oy_patch 20 --skip 7 8 9

dat_2d/data2-d1-192.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00242/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00242/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 192 --y_patch_size 192 --nx_patch 1 --ny_patch 1 --ox_patch 20 --oy_patch 20 --skip 7 8 9

makeData2:
	# make dat_2d/data2-t1-64.h5 -B
	# make dat_2d/data2-t1-192.h5 -B
	make dat_2d/data2-t2-64.h5 -B
	make dat_2d/data2-t2-192.h5 -B
	# make dat_2d/data2-d1-64.h5 -B
	# make dat_2d/data2-d1-192.h5 -B

# Training
train2:
	rm -rf models_2d/qc2
	python3.6 ./python/CTP_main_2d.py train qc2 --model EGNET --n_epochs 10 --lr 5.0e-3 --device cuda:0 --train_file dat_2d/data2-t1-64.h5 --dev_file dat_2d/data2-d1-64.h5 --batch_size 64 --half 1 --seed 10

train2-t2:
	rm -rf models_2d/qc2-t2
	python3.6 ./python/CTP_main_2d.py train qc2-t2 --model EGNET --n_epochs 10 --lr 5.0e-3 --device cuda:0 --train_file dat_2d/data2-t2-64.h5 --dev_file dat_2d/data2-d1-64.h5 --batch_size 64 --half 1 --seed 10

train2-t3:
	rm -rf models_2d/qc2-t3
	python3.6 ./python/CTP_main_2d.py train qc2-t3 --model EGNET --n_epochs 2 --lr 5.0e-3 --device cuda:0 --train_file_list par/train_2d_file.txt --dev_file dat_2d/data2-d1-64.h5 --batch_size 64 --half 1 --seed 10

# Display
display2-train2-t1-f%:
	Window3d n3=1 f3=$* < dat_2d/data2-t1-192.h5_labels_train.H > t0.H
	echo "d1=1.0 d2=1.0" >> t0.H
	Window3d n3=1 f3=$* < models_2d/qc2/qc2.h5_labels_train_2d_file0.H > t1.H
	Window3d n3=1 f3=$* < models_2d/qc2/qc2.h5_y_pred_train_2d_file0.H > t2.H
	Cat axis=3 t0.H t1.H t2.H | Grey color=j newclip=1 grid=y titles="True:Labels:Pred" | Xtpen pximaps=y &
	Add t1.H t0.H scale=1,-1 > diff.H
	Attr < diff.H

display2-train2-t2-f%:
	Window3d n3=1 f3=$* < dat_2d/data2-t2-192.h5_labels_train.H > t0.H
	echo "d1=1.0 d2=1.0" >> t0.H
	Window3d n3=1 f3=$* < models_2d/qc2-t2/qc2-t2.h5_labels_train_2d_file0.H > t1.H
	Window3d n3=1 f3=$* < models_2d/qc2-t2/qc2-t2.h5_y_pred_train_2d_file0.H > t2.H
	Cat axis=3 t0.H t1.H t2.H | Grey color=j newclip=1 grid=y titles="True:Labels:Pred" | Xtpen pximaps=y &
	Add t1.H t0.H scale=1,-1 > diff.H
	Attr < diff.H

display2-train2-t3-file1-f%:
	# File 1
	Window3d n3=1 f3=$* < dat_2d/data2-t1-192.h5_labels_train.H > t0.H
	echo "d1=1.0 d2=1.0" >> t0.H
	Window3d n3=1 f3=$* < models_2d/qc2-t3/qc2-t3.h5_labels_train_2d_file0.H > t1.H
	Window3d n3=1 f3=$* < models_2d/qc2-t3/qc2-t3.h5_y_pred_train_2d_file0.H > t2.H
	Cat axis=3 t0.H t1.H t2.H | Grey color=j newclip=1 grid=y titles="True:Labels:Pred" | Xtpen pximaps=y &
	Add t1.H t0.H scale=1,-1 > diff.H
	Attr < diff.H

display2-train2-t3-file2-f%:
	# File 1
	Window3d n3=1 f3=$* < dat_2d/data2-t2-192.h5_labels_train.H > t0.H
	echo "d1=1.0 d2=1.0" >> t0.H
	Window3d n3=1 f3=$* < models_2d/qc2-t3/qc2-t3.h5_labels_2d_train_file1.H > t1.H
	Window3d n3=1 f3=$* < models_2d/qc2-t3/qc2-t3.h5_y_pred_2d_train_file1.H > t2.H
	Cat axis=3 t0.H t1.H t2.H | Grey color=j newclip=1 grid=y titles="True:Labels:Pred" | Xtpen pximaps=y &
	# Add t1.H t0.H scale=1,-1 > diff.H
	# Attr < diff.H

display2-train2-d1-file2-f%:
	# File 1
	Window3d n3=1 f3=$* < dat_2d/data2-d1-192.h5_labels_train.H > t0.H
	echo "d1=1.0 d2=1.0" >> t0.H
	Window3d n3=1 f3=$* < models_2d/qc2-t3/qc2-t3.h5_labels_2d_dev_file0.H > t1.H
	Window3d n3=1 f3=$* < models_2d/qc2-t3/qc2-t3.h5_y_pred_2d_dev_file0.H > t2.H
	Cat axis=3 t0.H t1.H t2.H | Grey color=j newclip=1 grid=y titles="True:Labels:Pred" | Xtpen pximaps=y &
	# Add t1.H t0.H scale=1,-1 > diff.H
	# Attr < diff.H

################################# 1 slice with halos ###########################
# S00243 - 1 slice
dat_2d/data1-t1-halo1-64.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 64 --y_patch_size 64 --nx_patch 2 --ny_patch 2 --ox_patch 40 --oy_patch 30 --skip 0 1 2 4 5 6 7 8 9

dat_2d/data1-t1-halo2-64.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 2 --ny_patch 2 --ox_patch 40 --oy_patch 30 --skip 0 1 2 4 5 6 7 8 9 --halo 5

dat_2d/data1-t1-halo3-64.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 108 --y_patch_size 108 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 30 --skip 0 1 2 4 5 6 7 8 9

dat_2d/data1-t1-halo4-54.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 2 --ny_patch 2 --ox_patch 40 --oy_patch 30 --skip 0 2 4 5 6 7 8 9 --halo 5

dat_2d/data1-t1-halo4-108.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 108 --y_patch_size 108 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 30 --skip 0 2 4 5 6 7 8 9

makeTestComp:
	# make dat_2d/data1-t1-halo1-64.h5 -B
	# make dat_2d/data1-t1-halo2-64.h5 -B
	make dat_2d/data1-t1-halo4-54.h5 -B
	make dat_2d/data1-t1-halo4-108.h5 -B

testComp:
	# Add dat_2d/data1-t1-64.h5_data_train.H dat_2d/data1-t1-halo1-64.h5_data_train.H scale=1,-1 > diff1.H
	# Add dat_2d/data1-t1-64.h5_labels_train.H dat_2d/data1-t1-halo1-64.h5_labels_train.H scale=1,-1 > diff2.H
	# Attr < diff1.H
	# Attr < diff2.H
	# Window3d n1=1 f2=30 < dat_2d/data1-t1-halo1-64.h5_data_train.H | Grey color=g grid=y title="halo 1" | Xtpen &
	# Window3d n1=1 f2=30 < dat_2d/data1-t1-halo2-64.h5_data_train.H | Grey color=g grid=y title="halo 2" | Xtpen &
	# Window3d f1=0 n1=59 f2=0 n2=59 n3=1 f3=$* < dat_2d/data1-t1-halo1-64.h5_labels_train.H > t1.H
	# Labels
	# Window3d f1=5 n1=54 f2=5 n2=54 n3=1 f3=7 < dat_2d/data1-t1-halo4-54.h5_labels_train.H > t1.H
	# Window3d f1=54 n1=54 f2=54 n2=54 n3=1 f3=1 < dat_2d/data1-t1-halo4-108.h5_labels_train.H > t2.H
	# Cat axis=3 t1.H t2.H | Grey gainpanel=a grid=y color=j newclip=1 titles="Halo:No halo" | Xtpen pixmaps=y &
	# Add t1.H t2.H scale=1,-1 > diff.H
	# Attr < diff.H
	# Data
	Window3d n1=1 f1=30 f2=5 n2=54 f3=5 n3=54 n4=1 f4=7 < dat_2d/data1-t1-halo4-54.h5_data_train.H > t1.H
	Window3d n1=1 f1=30 f2=54 n2=54 f3=54 n3=54 n4=1 f4=1 < dat_2d/data1-t1-halo4-108.h5_data_train.H > t2.H
	Cat axis=3 t1.H t2.H | Grey gainpanel=a grid=y color=j newclip=1 titles="Halo:No halo" | Xtpen pixmaps=y &
	Add t1.H t2.H scale=1,-1 > diff.H
	Attr < diff.H

################################################################################
############################# Training without halos ###########################
################################################################################
# No halos
dat_2d/qc_data_train1_64.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 64 --y_patch_size 64 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 30 --skip 0 1 2 4 5 6 7 8 9

dat_2d/qc_data_train1_128.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 192 --y_patch_size 192 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 30 --skip 0 1 2 4 5 6 7 8 9

dat_2d/qc_data_dev1_64.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 64 --y_patch_size 64 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 30 --skip 0 1 3 4 5 6 7 8 9

dat_2d/qc_data_dev1_128.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 192 --y_patch_size 192 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 30 --skip 0 1 3 4 5 6 7 8 9

makeDatQc:
	make dat_2d/qc_data_train1_64.h5 -B
	make dat_2d/qc_data_train1_128.h5 -B
	make dat_2d/qc_data_dev1_64.h5 -B
	make dat_2d/qc_data_dev1_128.h5 -B

train2d_qc1:
	rm -rf models_2d/train2d_qc1
	python3.6 ./python/CTP_main_2d.py train train2d_qc1 --model EGNET --n_epochs 500 --lr 5.0e-3 --device cuda:0 --train_file dat_2d/qc_data_train1_64.h5 --dev_file dat_2d/qc_data_dev1_64.h5 --batch_size 32 --half 1 --seed 10

dispPred-train-noHalo:
	# 2d prediction
	Add models_2d/train2d_qc1/train2d_qc1.h5_y_pred_2d_train_file0.H models_2d/train2d_qc1/train2d_qc1.h5_y_pred_2d_halo_train_file0.H scale=1,-1 > diff1.H
	Add models_2d/train2d_qc1/train2d_qc1.h5_labels_2d_train_file0.H models_2d/train2d_qc1/train2d_qc1.h5_labels_2d_halo_train_file0.H scale=1,-1 > diff2.H
	Add models_2d/train2d_t1_m1/train2d_t1_m1.h5_labels_2d_train_file0.H models_2d/train2d_qc1/train2d_qc1.h5_labels_2d_train_file0.H scale=1,-1 > diff3.H
	Attr < diff1.H
	Attr < diff2.H
	Attr < diff3.H
	Window3d f1=0 < dat_2d/qc_data_train1_128.h5_labels_train.H > t0.H
	Window3d f1=0 < models_2d/train2d_qc1/train2d_qc1.h5_labels_2d_train_file0.H > t1.H
	Window3d f1=0 < models_2d/train2d_qc1/train2d_qc1.h5_y_pred_2d_train_file0.H > t2.H
	Window3d f1=0 < models_2d/train2d_t1_m1/train2d_t1_m1.h5_labels_2d_train_file0.H > t3.H
	Window3d f1=0 < models_2d/train2d_t1_m1/train2d_t1_m1.h5_y_pred_2d_train_file0.H > t4.H
	Add t2.H t4.H scale=1,-1 > diff4.H
	Attr < diff4.H
	Add t0.H t1.H scale=1,-1 > diff5.H
	Attr < diff5.H
	Cat axis=3 t0.H t1.H t2.H | Grey color=j newclip=1 titles="Labels:Labels out:Pred" | Xtpen pixmaps=y &

dispPred-dev-noHalo:
	# 2d prediction
	Add models_2d/train2d_qc1/train2d_qc1.h5_y_pred_2d_dev_file0.H models_2d/train2d_qc1/train2d_qc1.h5_y_pred_2d_halo_dev_file0.H scale=1,-1 > diff1.H
	Add models_2d/train2d_qc1/train2d_qc1.h5_labels_2d_dev_file0.H models_2d/train2d_qc1/train2d_qc1.h5_labels_2d_halo_dev_file0.H scale=1,-1 > diff2.H
	Add models_2d/train2d_t1_m1/train2d_t1_m1.h5_labels_2d_dev_file0.H models_2d/train2d_qc1/train2d_qc1.h5_labels_2d_dev_file0.H scale=1,-1 > diff3.H
	Attr < diff1.H
	Attr < diff2.H
	Attr < diff3.H
	Window3d f1=0 < dat_2d/qc_data_dev1_128.h5_labels_train.H > t0.H
	Window3d f1=0 < models_2d/train2d_qc1/train2d_qc1.h5_labels_2d_dev_file0.H > t1.H
	Window3d f1=0 < models_2d/train2d_qc1/train2d_qc1.h5_y_pred_2d_dev_file0.H > t2.H
	Window3d f1=0 < models_2d/train2d_t1_m1/train2d_t1_m1.h5_labels_2d_dev_file0.H > t3.H
	Window3d f1=0 < models_2d/train2d_t1_m1/train2d_t1_m1.h5_y_pred_2d_dev_file0.H > t4.H
	Add t2.H t4.H scale=1,-1 > diff4.H
	Attr < diff4.H
	Add t0.H t1.H scale=1,-1 > diff5.H
	Attr < diff5.H
	Cat axis=3 t0.H t1.H t2.H | Grey color=j newclip=1 titles="Labels:Labels out:Pred" grid=y | Xtpen pixmaps=y &

################################################################################
############################# Training with halos ##############################
################################################################################
# No halos
dat_2d/qc_data_train1_64_halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 30 --skip 0 1 2 4 5 6 7 8 9 --halo 5

dat_2d/qc_data_train1_128_halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 162 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 30 --skip 0 1 2 4 5 6 7 8 9

dat_2d/qc_data_dev1_64_halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 30 --skip 0 1 3 4 5 6 7 8 9 --halo 5

dat_2d/qc_data_dev1_128_halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 162 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 30 --skip 0 1 3 4 5 6 7 8 9

makeDatQcHalo:
	make dat_2d/qc_data_train1_64_halo.h5 -B
	make dat_2d/qc_data_train1_128_halo.h5 -B
	make dat_2d/qc_data_dev1_64_halo.h5 -B
	make dat_2d/qc_data_dev1_128_halo.h5 -B

train2d_qc1_halo:
	rm -rf models_2d/train2d_qc1_halo
	python3.6 ./python/CTP_main_2d.py train train2d_qc1_halo --model EGNET --n_epochs 500 --lr 5.0e-3 --device cuda:0 --train_file dat_2d/qc_data_train1_64_halo.h5 --dev_file dat_2d/qc_data_dev1_64_halo.h5 --batch_size 32 --half 1 --seed 10

dispPred-train-halo:
	# 2d prediction
	# Window3d n1=162 f1=0 n2=162 f2=0 < models_2d/train2d_qc1_halo/train2d_qc1_halo.h5_labels_2d_train_file0.H > t1.H
	# Window3d n1=162 f1=0 n2=162 f2=0 < dat_2d/qc_data_train1_128_halo.h5_labels_train.H > t2.H
	# Window3d n1=162 f1=0 n2=162 f2=0 < models_2d/train2d_qc1_halo/train2d_qc1_halo.h5_y_pred_2d_train_file0.H > t3.H
	# Add t1.H t2.H scale=1,-1 > diff1.H
	# Attr < diff1.H
	Cat axis=3 t1.H t2.H t3.H | Grey color=j newclip=1 grid=y titles="Rapdid:Labels out:Pred" | Xtpen pixmaps=y &
	Window3d f1=0 < models_2d/train2d_qc1_halo/train2d_qc1_halo.h5_labels_2d_halo_train_file0.H > t4.H
	Window3d f1=0 < models_2d/train2d_qc1_halo/train2d_qc1_halo.h5_y_pred_2d_halo_train_file0.H > t5.H
	Cat axis=3 t4.H t5.H | Grey color=j newclip=1 grid=y titles="Labels:Pred" | Xtpen pixmaps=y &

dispPred-dev-halo:
	# 2d prediction
	Window3d n1=162 f1=0 n2=162 f2=0 < models_2d/train2d_qc1_halo/train2d_qc1_halo.h5_labels_2d_dev_file0.H > t1.H
	Window3d n1=162 f1=0 n2=162 f2=0 < dat_2d/qc_data_dev1_128_halo.h5_labels_train.H > t2.H
	Window3d n1=162 f1=0 n2=162 f2=0 < models_2d/train2d_qc1_halo/train2d_qc1_halo.h5_y_pred_2d_dev_file0.H > t3.H
	Add t1.H t2.H scale=1,-1 > diff1.H
	Attr < diff1.H
	Cat axis=3 t1.H t2.H t3.H | Grey color=j newclip=1 grid=y titles="Rapdid halo:Labels out halo:Pred halo" | Xtpen pixmaps=y &
	# Window3d f1=0 < models_2d/train2d_qc1_halo/train2d_qc1_halo.h5_labels_2d_dev_file0.H > t4.H
	# Window3d f1=0 < models_2d/train2d_qc1_halo/train2d_qc1_halo.h5_y_pred_2d_dev_file0.H > t5.H
	# Cat axis=3 t4.H t5.H | Grey color=j newclip=1 grid=y titles="Labels:Pred" | Xtpen pixmaps=y &

# compare halo/no halo
compHalo:
	# Train
	# Window3d n1=162 f1=0 n2=162 f2=0 < models_2d/train2d_qc1/train2d_qc1.h5_labels_2d_train_file0.H > t1.H
	# Window3d n1=162 f1=0 n2=162 f2=0 < models_2d/train2d_qc1/train2d_qc1.h5_y_pred_2d_train_file0.H > t1p.H
	# Window3d n1=162 f1=0 n2=162 f2=0 < models_2d/train2d_qc1_halo/train2d_qc1_halo.h5_labels_2d_train_file0.H > t2.H
	# Window3d n1=162 f1=0 n2=162 f2=0 < models_2d/train2d_qc1_halo/train2d_qc1_halo.h5_y_pred_2d_train_file0.H > t2p.H
	# Add t1.H t2.H scale=1,-1 > diff1.H
	# Attr < diff1.H
	# Cat axis=3 t1.H t2.H t1p.H t2p.H | Grey color=j newclip=1 grid=y titles="labels:labels halo:pred:pred halo" gainpanel=a bclip=0 eclip=400 | Xtpen pixmaps=y &
	# Dev
	Window3d n1=162 f1=0 n2=162 f2=0 < models_2d/train2d_qc1/train2d_qc1.h5_labels_2d_dev_file0.H > t1.H
	Window3d n1=162 f1=0 n2=162 f2=0 < models_2d/train2d_qc1/train2d_qc1.h5_y_pred_2d_dev_file0.H > t1p.H
	Window3d n1=162 f1=0 n2=162 f2=0 < models_2d/train2d_qc1_halo/train2d_qc1_halo.h5_labels_2d_dev_file0.H > t2.H
	Window3d n1=162 f1=0 n2=162 f2=0 < models_2d/train2d_qc1_halo/train2d_qc1_halo.h5_y_pred_2d_dev_file0.H > t2p.H
	Add t1.H t2.H scale=1,-1 > diff1.H
	Attr < diff1.H
	Cat axis=3 t1.H t1p.H t2p.H | Grey color=j newclip=1 grid=y titles="labels:pred:pred halo" gainpanel=a bclip=0 eclip=400 | Xtpen pixmaps=y &

################################################################################
############################# QC memory conv1d #################################
################################################################################
eegg_qc_memory_v1:
	rm -rf models_2d/eegg_qc_memory_v1
	python3.6 ./python/CTP_main_2d.py train eegg_qc_memory_v1 --model EGNET --n_epochs 100 --lr 1.0e-3 --device cuda:0 --train_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 64 --half 1 --seed 10

eegg_qc_memory_v2:
	rm -rf models_2d/eegg_qc_memory_v2
	python3.6 ./python/CTP_main_2d.py train eegg_qc_memory_v2 --model EGNET_batch --n_epochs 100 --lr 1.0e-3 --device cuda:1 --train_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 64 --half 1 --seed 10
