################################################################################
################################ 1 axial slice #################################
################################################################################

################################## Test 1 ######################################
train2d_t1_m1:
	rm -rf models_2d/train2d_t1_m1
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m1 --model EGNET --n_epochs 500 --lr 5.0e-3 --device cuda:0 --train_file dat_2d/S00243-dat-t1-v1-64.h5 --dev_file dat_2d/S00243-dat-t1-v2-64.h5 --batch_size 32 --half 1 --seed 10

train2d_t1_m1_res:
	# Train
	Cp dat_2d/S00243-dat-t1-v1-192.h5_labels_train.H t0t.H
	echo "d1=1.0 d2=1.0" >> t0t.H
	Cp models_2d/train2d_t1_m1/train2d_t1_m1.h5_labels_2d_train_file0.H t1t.H
	Cp models_2d/train2d_t1_m1/train2d_t1_m1.h5_y_pred_2d_train_file0.H t2t.H
	Cat axis=3 t1t.H t2t.H | Grey color=j newclip=1 grid=y titles="Labels (train):Pred (train)" | Xtpen pximaps=y &
	# Dev
	Cp dat_2d/S00243-dat-t1-v2-192.h5_labels_train.H t0d.H
	echo "d1=1.0 d2=1.0" >> t0d.H
	Cp models_2d/train2d_t1_m1/train2d_t1_m1.h5_labels_2d_dev_file0.H t1d.H
	Cp models_2d/train2d_t1_m1/train2d_t1_m1.h5_y_pred_2d_dev_file0.H t2d.H
	Cat axis=3 t1d.H t2d.H | Grey color=j newclip=1 grid=y titles="Labels (dev):Pred (dev)" | Xtpen pximaps=y &
	# QC
	Add t1t.H t0t.H scale=1,-1 > t_diff.H
	Add t1d.H t0d.H scale=1,-1 > d_diff.H

train2d_t1_m1_stat:
	# Data
	Histogram dinterv=10 min=0 max=150 < dat_2d/S00243-dat-t1-v1-64.h5_data_train.H | Scale > t1.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/S00243-dat-t1-v2-64.h5_data_train.H | Scale > t2.H
	Cat axis=2 t1.H t2.H | Scale | Graph grid=y min1=0.0 max1=150 min2=0.0 max2=1.0 legend=y curvelabel="data train:data dev" legendloc=tr | Xtpen &
	# Labels
	Histogram dinterv=10 min=0 max=400 < dat_2d/S00243-dat-t1-v1-64.h5_labels_train.H | Scale > t1.H
	Histogram dinterv=10 min=0 max=400 < dat_2d/S00243-dat-t1-v2-64.h5_labels_train.H | Scale > t2.H
	Cat axis=2 t1.H t2.H | Scale | Graph grid=y min1=0.0 max1=150 min2=0.0 max2=1.0 legend=y curvelabel="labels train:labels dev" legendloc=tr | Xtpen &

################################## Test 2 ######################################
train2d_t1_m2:
	rm -rf models_2d/train2d_t1_m2
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m2 --model EGNET --n_epochs 500 --lr 5.0e-3 --device cuda:0 --train_file dat_2d/S00243-dat-t1-v1-64.h5 --dev_file dat_2d/S00243-dat-t1-v4-64.h5 --batch_size 32 --half 1 --seed 10

train2d_t1_m2_res%:
	# Train
	Cp dat_2d/S00243-dat-t1-v1-192.h5_labels_train.H t0t.H
	echo "d1=1.0 d2=1.0" >> t0t.H
	Cp models_2d/train2d_t1_m2/train2d_t1_m2.h5_labels_2d_train_file0.H t1t.H
	Cp models_2d/train2d_t1_m2/train2d_t1_m2.h5_y_pred_2d_train_file0.H t2t.H
	Cat axis=3 t1t.H t2t.H | Grey color=j newclip=1 grid=y titles="Labels (train):Pred (train)" | Xtpen pximaps=y &
	# Dev
	Window3d n3=1 f3=$* < dat_2d/S00243-dat-t1-v3-192.h5_labels_train.H > t0d.H
	echo "d1=1.0 d2=1.0" >> t0d.H
	Window3d n3=1 f3=$* < models_2d/train2d_t1_m2/train2d_t1_m2.h5_labels_2d_dev_file0.H > t1d.H
	Window3d n3=1 f3=$* < models_2d/train2d_t1_m2/train2d_t1_m2.h5_y_pred_2d_dev_file0.H > t2d.H
	Cat axis=3 t1d.H t2d.H | Grey color=j newclip=1 grid=y titles="Labels (dev):Pred (dev)" | Xtpen pximaps=y &
	# QC
	# Add t1t.H t0t.H scale=1,-1 > t_diff.H
	# Add t1d.H t0d.H scale=1,-1 > d_diff.H

train2d_t1_m2_stat:
	# Data
	Histogram dinterv=10 min=0 max=150 < dat_2d/S00243-dat-t1-v1-64.h5_data_train.H | Scale > t1.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/S00243-dat-t1-v4-64.h5_data_train.H | Scale > t2.H
	Cat axis=2 t1.H t2.H | Scale | Graph grid=y min1=0.0 max1=150 min2=0.0 max2=1.0 legend=y curvelabel="Train:Dev" | Xtpen &
	# Labels
	Histogram dinterv=10 min=0 max=400 < dat_2d/S00243-dat-t1-v1-64.h5_labels_train.H | Scale > t1.H
	Histogram dinterv=10 min=0 max=400 < dat_2d/S00243-dat-t1-v4-64.h5_labels_train.H | Scale > t2.H
	Cat axis=2 t1.H t2.H | Scale | Graph grid=y min1=0.0 max1=150 min2=0.0 max2=1.0 legend=y curvelabel="labels train:labels dev" legendloc=tr | Xtpen &

################################## Test 3 ######################################
train2d_t1_m3:
	rm -rf models_2d/train2d_t1_m3
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m3 --model EGNET --n_epochs 500 --lr 5.0e-3 --device cuda:0 --train_file dat_2d/S00243-dat-t1-v1-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 32 --half 1 --seed 10

train2d_t1_m3_res%:
	# Train
	# Cp dat_2d/S00243-dat-t1-v1-192.h5_labels_train.H t0t.H
	# echo "d1=1.0 d2=1.0" >> t0t.H
	# Cp models_2d/train2d_t1_m3/train2d_t1_m3.h5_labels_2d_train_file0.H t1t.H
	# Cp models_2d/train2d_t1_m3/train2d_t1_m3.h5_y_pred_2d_train_file0.H t2t.H
	# Cat axis=3 t1t.H t2t.H | Grey color=j newclip=1 grid=y titles="Labels (train):Pred (train)" | Xtpen pximaps=y &
	# Dev
	Window3d n3=1 f3=$* < dat_2d/S00243-dat-t1-v3-192.h5_labels_train.H > t0d.H
	echo "d1=1.0 d2=1.0" >> t0d.H
	Window3d n3=1 f3=$* < models_2d/train2d_t1_m3/train2d_t1_m3.h5_labels_2d_dev_file0.H > t1d.H
	Window3d n3=1 f3=$* < models_2d/train2d_t1_m3/train2d_t1_m3.h5_y_pred_2d_dev_file0.H > t2d.H
	Cat axis=3 t1d.H t2d.H | Grey color=j newclip=1 grid=y titles="Labels (dev):Pred (dev)" | Xtpen pximaps=y &
	# QC
	# Add t1t.H t0t.H scale=1,-1 > t_diff.H
	# Add t1d.H t0d.H scale=1,-1 > d_diff.H

train2d_t1_m3_stat:
	# Data
	Histogram dinterv=10 min=0 max=150 < dat_2d/S00243-dat-t1-v1-64.h5_data_train.H | Scale > t1.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/S00242-dat-t1-full-64.h5_data_train.H | Scale > t2.H
	Cat axis=2 t1.H t2.H | Scale | Graph grid=y min1=0.0 max1=150 min2=0.0 max2=1.0 legend=y curvelabel="Train:Dev" | Xtpen &
	# Labels
	Histogram dinterv=10 min=0 max=400 < dat_2d/S00243-dat-t1-v1-64.h5_labels_train.H | Scale > t1.H
	Histogram dinterv=10 min=0 max=400 < dat_2d/S00242-dat-t1-full-64.h5_labels_train.H | Scale > t2.H
	Cat axis=2 t1.H t2.H | Scale | Graph grid=y min1=0.0 max1=150 min2=0.0 max2=1.0 legend=y curvelabel="labels train:labels dev" legendloc=tr | Xtpen &

################################## Test 4 ######################################
train2d_t1_m4:
	rm -rf models_2d/train2d_t1_m4
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m4 --model EGNET --n_epochs 500 --lr 5.0e-3 --device cuda:0 --train_file dat_2d/S00242-dat-t1-v1-64.h5 --dev_file dat_2d/S00243-dat-t1-v4-64.h5 --batch_size 32 --half 1 --seed 10

train2d_t1_m4_res%:
	# Train
	# Cp dat_2d/S00242-dat-t1-v1-192.h5_labels_train.H t0t.H
	# echo "d1=1.0 d2=1.0" >> t0t.H
	# Cp models_2d/train2d_t1_m4/train2d_t1_m4.h5_labels_2d_train_file0.H t1t.H
	# Cp models_2d/train2d_t1_m4/train2d_t1_m4.h5_y_pred_2d_train_file0.H t2t.H
	# Cat axis=3 t1t.H t2t.H | Grey color=j newclip=1 grid=y titles="Labels (train):Pred (train)" | Xtpen pximaps=y &
	# Dev
	Window3d n3=1 f3=$* < dat_2d/S00243-dat-t1-v4-192.h5_labels_train.H > t0d.H
	echo "d1=1.0 d2=1.0" >> t0d.H
	Window3d n3=1 f3=$* < models_2d/train2d_t1_m4/train2d_t1_m4.h5_labels_2d_dev_file0.H > t1d.H
	Window3d n3=1 f3=$* < models_2d/train2d_t1_m4/train2d_t1_m4.h5_y_pred_2d_dev_file0.H > t2d.H
	Cat axis=3 t0d.H t1d.H t2d.H | Grey color=j newclip=1 bclip=0 eclip=400 grid=y titles="Rapid:Labels (dev):Pred (dev)" | Xtpen pximaps=y &
	# QC
	# Add t1t.H t0t.H scale=1,-1 > t_diff.H
	# Add t1d.H t0d.H scale=1,-1 > d_diff.H

train2d_t1_m4_stat:
	Histogram dinterv=10 min=0 max=150 < dat_2d/S00242-dat-t1-v1-64.h5_data_train.H | Scale > t1.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/S00243-dat-t1-v4-64.h5_data_train.H | Scale > t2.H
	Cat axis=2 t1.H t2.H | Scale | Graph grid=y min1=0.0 max1=150 min2=0.0 max2=1.0 legend=y curvelabel="Train:Dev" | Xtpen &
	# Labels
	Histogram dinterv=10 min=0 max=400 < dat_2d/S00242-dat-t1-v1-64.h5_labels_train.H | Scale > t1.H
	Histogram dinterv=10 min=0 max=400 < dat_2d/S00243-dat-t1-v4-64.h5_labels_train.H | Scale > t2.H
	Cat axis=2 t1.H t2.H | Scale | Graph grid=y min1=0.0 max1=150 min2=0.0 max2=1.0 legend=y curvelabel="labels train:labels dev" legendloc=tr | Xtpen &

################################## Test 5 ######################################
# Training on full head S00243
train2d_t1_m5-v1:
	rm -rf models_2d/train2d_t1_m5-v1
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v1 --model EGNET --n_epochs 100 --lr 1.0e-5 --device cuda:0 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 64 --half 1 --seed 10

train2d_t1_m5-v2:
	rm -rf models_2d/train2d_t1_m5-v2
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v2 --model EGNET --n_epochs 100 --lr 5.0e-5 --device cuda:1 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 64 --half 1 --seed 10

train2d_t1_m5-v3:
	rm -rf models_2d/train2d_t1_m5-v3
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v3 --model EGNET --n_epochs 100 --lr 1.0e-4 --device cuda:2 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 64 --half 1 --seed 10

train2d_t1_m5-v4:
	rm -rf models_2d/train2d_t1_m5-v4
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v4 --model EGNET --n_epochs 100 --lr 5.0e-4 --device cuda:3 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 64 --half 1 --seed 10

train2d_t1_m5-v5:
	rm -rf models_2d/train2d_t1_m5-v5
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v5 --model EGNET --n_epochs 100 --lr 1.0e-3 --device cuda:0 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 64 --half 1 --seed 10

train2d_t1_m5-v6:
	rm -rf models_2d/train2d_t1_m5-v6
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v6 --model EGNET --n_epochs 100 --lr 2.5e-3 --device cuda:1 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 64 --half 1 --seed 10

train2d_t1_m5-v7:
	rm -rf models_2d/train2d_t1_m5-v7
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v7 --model EGNET --n_epochs 100 --lr 2.5e-3 --device cuda:1 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 128 --half 1 --seed 10

train2d_t1_m5-v8:
	rm -rf models_2d/train2d_t1_m5-v8
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v8 --model EGNET --n_epochs 500 --lr 5e-3 --device cuda:0 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 160 --half 1 --seed 10

train2d_t1_m5-v9:
	rm -rf models_2d/train2d_t1_m5-v9
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v9 --model EGNET --n_epochs 500 --lr 7.5e-3 --device cuda:1 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 160 --half 1 --seed 10

train2d_t1_m5-v10:
	rm -rf models_2d/train2d_t1_m5-v10
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v10 --model EGNET --n_epochs 500 --lr 1.0e-2 --device cuda:2 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 160 --half 1 --seed 10

train2d_t1_m5-v11:
	rm -rf models_2d/train2d_t1_m5-v11
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v11 --model EGNET --n_epochs 500 --lr 1.0e-2 --device cuda:0 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 160 --half 1 --seed 10	--lr_decay step --step_size 10 --decay_gamma 0.99

train2d_t1_m5-v12:
	rm -rf models_2d/train2d_t1_m5-v12
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v12 --model EGNET --n_epochs 500 --lr 1.0e-2 --device cuda:1 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 160 --half 1 --seed 10	--lr_decay step --step_size 50 --decay_gamma 0.99

train2d_t1_m5-v13:
	rm -rf models_2d/train2d_t1_m5-v13
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v13 --model EGNET --n_epochs 500 --lr 1.0e-2 --device cuda:2 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 160 --half 1 --seed 10	--lr_decay step --step_size 10 --decay_gamma 0.9

train2d_t1_m5-v14:
	rm -rf models_2d/train2d_t1_m5-v14
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v14 --model EGNET --n_epochs 500 --lr 1.0e-2 --device cuda:3 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 160 --half 1 --seed 10	--lr_decay step --step_size 100 --decay_gamma 0.99

train2d_t1_m5-v15:
	rm -rf models_2d/train2d_t1_m5-v15
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v15 --model EGNET --n_epochs 500 --lr 0.2e-2 --device cuda:0 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 160 --half 1 --seed 10 --lr_decay step --step_size 20 --decay_gamma 0.99

train2d_t1_m5-v16:
	rm -rf models_2d/train2d_t1_m5-v16
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v16 --model EGNET --n_epochs 500 --lr 0.1e-2 --device cuda:1 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 160 --half 1 --seed 10 --lr_decay step --step_size 10 --decay_gamma 0.99

train2d_t1_m5-v17:
	rm -rf models_2d/train2d_t1_m5-v17
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v17 --model EGNET --n_epochs 500 --lr 0.1e-2 --device cuda:2 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 160 --half 1 --seed 10 --lr_decay step --step_size 5 --decay_gamma 0.99

train2d_t1_m5-v18:
	rm -rf models_2d/train2d_t1_m5-v18
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v18 --model EGNET --n_epochs 500 --lr 0.5e-3 --device cuda:3 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 160 --half 1 --seed 10	--lr_decay step --step_size 5 --decay_gamma 0.999

train2d_t1_m5-v19:
	rm -rf models_2d/train2d_t1_m5-v19
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v19 --model EGNET --n_epochs 500 --lr 0.2e-2 --device cuda:0 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 160 --half 1 --seed 10 --lr_decay step --step_size 10 --decay_gamma 0.99

train2d_t1_m5-v20:
	rm -rf models_2d/train2d_t1_m5-v20
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v20 --model EGNET --n_epochs 500 --lr 0.3e-2 --device cuda:1 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 160 --half 1 --seed 10 --lr_decay step --step_size 20 --decay_gamma 0.99

train2d_t1_m5-v21:
	rm -rf models_2d/train2d_t1_m5-v21
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v21 --model EGNET --n_epochs 500 --lr 0.4e-2 --device cuda:2 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 160 --half 1 --seed 10 --lr_decay step --step_size 20 --decay_gamma 0.99

train2d_t1_m5-v22:
	rm -rf models_2d/train2d_t1_m5-v22
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m5-v22 --model EGNET --n_epochs 500 --lr 0.4e-2 --device cuda:3 --train_file dat_2d/S00243-dat-t1-v4-64.h5 --dev_file dat_2d/S00242-dat-t1-full-64.h5 --batch_size 160 --half 1 --seed 10 --lr_decay step --step_size 40 --decay_gamma 0.99		

train2d_t1_m5_res%:
	# Train
	# Window3d n3=1 f3=$* < dat_2d/S00243-dat-t1-v4-192.h5_labels_train.H > t0t.H
	# echo "d1=1.0 d2=1.0" >> t0t.H
	# Window3d n3=1 f3=$* < models_2d/train2d_t1_m5-v8/train2d_t1_m5-v8.h5_labels_2d_train_file0.H > t1t.H
	# Window3d n3=1 f3=$* < models_2d/train2d_t1_m5-v8/train2d_t1_m5-v8.h5_y_pred_2d_train_file0.H > t2t.H
	# Cat axis=3 t1t.H t2t.H | Grey color=j newclip=1 grid=y titles="Labels (train):Pred (train)" | Xtpen pximaps=y &
	# Dev
	Window3d n3=1 f3=$* < dat_2d/S00242-dat-t1-full-192.h5_labels_train.H > t0d.H
	echo "d1=1.0 d2=1.0" >> t0d.H
	Window3d n3=1 f3=$* < models_2d/train2d_t1_m5-v9/train2d_t1_m5-v9.h5_labels_2d_dev_file0.H > t1d.H
	Window3d n3=1 f3=$* < models_2d/train2d_t1_m5-v9/train2d_t1_m5-v9.h5_y_pred_2d_dev_file0.H > t2d.H
	Cat axis=3 t0d.H t1d.H t2d.H | Grey color=j newclip=1 bclip=0 eclip=400 grid=y titles="Rapid:Labels (dev):Pred (dev)" | Xtpen pximaps=y &
	# QC
	# Add t1t.H t0t.H scale=1,-1 > t_diff.H
	# Add t1d.H t0d.H scale=1,-1 > d_diff.H

train2d_t1_m5_stat:
	Histogram dinterv=10 min=0 max=150 < dat_2d/S00243-dat-t1-v4-64.h5_data_train.H | Scale > t1.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/S00242-dat-t1-full-64.h5_data_train.H | Scale > t2.H
	Cat axis=2 t1.H t2.H | Scale | Graph grid=y min1=0.0 max1=150 min2=0.0 max2=1.0 legend=y curvelabel="Train:Dev" | Xtpen &

################################## Test 6 ######################################
# Training on full head S00243 minus 1 slice
train2d_t1_m6:
	rm -rf models_2d/train2d_t1_m6
	python3.6 ./python/CTP_main_2d.py train train2d_t1_m6 --model EGNET --n_epochs 250 --lr 5.0e-3 --device cuda:0 --train_file dat_2d/S00243-dat-t1-v5-64.h5 --dev_file dat_2d/S00243-dat-t1-v4-64.h5 --batch_size 32 --half 1 --seed 10

train2d_t1_m6_res%:
	# Train
	# Window3d n3=1 f3=$* < dat_2d/S00243-dat-t1-v5-192.h5_labels_train.H > t0t.H
	# echo "d1=1.0 d2=1.0" >> t0t.H
	# Window3d n3=1 f3=$* < models_2d/train2d_t1_m6/train2d_t1_m6.h5_labels_2d_train_file0.H > t1t.H
	# Window3d n3=1 f3=$* < models_2d/train2d_t1_m6/train2d_t1_m6.h5_y_pred_2d_train_file0.H > t2t.H
	# Cat axis=3 t0t.H t1t.H t2t.H | Grey color=j newclip=1 grid=y titles="Rapid:Labels (train):Pred (train)" | Xtpen pximaps=y &
	# Dev
	Window3d n3=1 f3=$* < dat_2d/S00243-dat-t1-v4-192.h5_labels_train.H > t0d.H
	echo "d1=1.0 d2=1.0" >> t0d.H
	Window3d n3=1 f3=$* < models_2d/train2d_t1_m6/train2d_t1_m6.h5_labels_2d_dev_file0.H > t1d.H
	Window3d n3=1 f3=$* < models_2d/train2d_t1_m6/train2d_t1_m6.h5_y_pred_2d_dev_file0.H > t2d.H
	Cat axis=3 t0d.H t1d.H t2d.H | Grey color=j newclip=1 bclip=0 eclip=400 grid=y titles="Rapid:Labels (dev):Pred (dev)" | Xtpen pximaps=y &
	# QC
	# Add t1t.H t0t.H scale=1,-1 > t_diff.H
	# Add t1d.H t0d.H scale=1,-1 > d_diff.H
