################################################################################
################################ 1 axial slice #################################
################################################################################

################################## Test 1 ######################################
eegg_train1_m1:
	rm -rf models_2d/eegg_train1_m1
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m1 --model EGNET --n_epochs 800 --lr 2.0e-3 --device cuda:0 --train_file dat_2d/eegg-dat-S00243-t1-v1-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00243-t1-v2-patch-halo.h5 --batch_size 32 --half 1 --seed 10

eegg_train1_m1_v1:
	rm -rf models_2d/eegg_train1_m1_v1
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m1_v1 --model EGNET --n_epochs 400 --lr 5.0e-3 --device cuda:0 --train_file dat_2d/eegg-dat-S00243-t1-v1-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00243-t1-v2-patch-halo.h5 --batch_size 32 --half 1 --seed 10

eegg_train1_m1_result:
	# Data QC
	# Window3d n1=1 f1=30 < dat_2d/eegg-dat-S00243-t1-v1-total-halo.h5_data_train.H | Grey color=g grid=y newclip=1 bclip=0 eclip=160 | Xtpen pixmaps=y &
	# Window3d n1=1 f1=30 < dat_2d/eegg-dat-S00243-t1-v1-full-halo.h5_data_train.H | Grey color=g grid=y newclip=1 bclip=0 eclip=160 | Xtpen pixmaps=y &
	# Window3d f1=0 < dat_2d/eegg-dat-S00243-t1-v1-total-halo.h5_labels_train.H | Grey color=j grid=y newclip=1 bclip=0 eclip=400 | Xtpen pixmaps=y &
	# Window3d f1=0 < dat_2d/eegg-dat-S00243-t1-v1-full-halo.h5_labels_train.H | Grey color=j grid=y newclip=1 bclip=0 eclip=400 | Xtpen pixmaps=y &
	# Train
	Cp dat_2d/eegg-dat-S00243-t1-v1-patch-halo.h5_labels_train.H t0t.H
	echo "d1=1.0 d2=1.0" >> t0t.H
	Cp models_2d/eegg_train1_m1/eegg_train1_m1.h5_labels_2d_train_file0.H t1t.H
	Cp models_2d/eegg_train1_m1/eegg_train1_m1.h5_y_pred_2d_train_file0.H t2t.H
	Cat axis=3 t1t.H t2t.H | Grey color=j newclip=1 grid=y titles="Labels (train):Pred (train)" | Xtpen pximaps=y &
	# Dev
	Cp dat_2d/eegg-dat-S00243-t1-v2-patch-halo.h5_labels_train.H t0d.H
	echo "d1=1.0 d2=1.0" >> t0d.H
	Cp models_2d/eegg_train1_m1/eegg_train1_m1.h5_labels_2d_dev_file0.H t1d.H
	Cp models_2d/eegg_train1_m1/eegg_train1_m1.h5_y_pred_2d_dev_file0.H t2d.H
	Cat axis=3 t1d.H t2d.H | Grey color=j newclip=1 grid=y titles="Labels (dev):Pred (dev)" | Xtpen pximaps=y &

################################## Test 2 ######################################
eegg_train1_m2_v1:
	rm -rf models_2d/eegg_train1_m2_v1
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m2_v1 --model EGNET --n_epochs 700 --lr 2.0e-3 --device cuda:0 --train_file dat_2d/eegg-dat-S00243-t1-v1-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --batch_size 64 --half 1 --seed 10

eegg_train1_m2_result-s%:
	# Train
	Cp dat_2d/eegg-dat-S00243-t1-v1-patch-halo.h5_labels_train.H t0t.H
	echo "d1=1.0 d2=1.0" >> t0t.H
	Cp models_2d/eegg_train1_m1/eegg_train1_m1.h5_labels_2d_train_file0.H t1t.H
	Cp models_2d/eegg_train1_m1/eegg_train1_m1.h5_y_pred_2d_train_file0.H t2t.H
	Cat axis=3 t1t.H t2t.H | Grey color=j newclip=1 grid=y titles="Labels (train):Pred (train)" | Xtpen pximaps=y &
	# Dev
	Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00243-t1-v3-full-halo.h5_labels_train.H > t0d.H
	echo "d1=1.0 d2=1.0" >> t0d.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m2_v1/eegg_train1_m2_v1.h5_labels_2d_dev_file0.H > t1d.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m2_v1/eegg_train1_m2_v1.h5_y_pred_2d_dev_file0.H > t2d.H
	Cat axis=3 t1d.H t2d.H | Grey color=j newclip=1 grid=y titles="Labels (dev):Pred (dev)" | Xtpen pximaps=y &

################################## Test 3 ######################################
eegg_train1_m3_v1:
	rm -rf models_2d/eegg_train1_m3_v1
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m3_v1 --model EGNET --n_epochs 700 --lr 2.0e-3 --device cuda:2 --train_file dat_2d/eegg-dat-S00243-t1-v1-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 64 --half 0 --seed 10

eegg_train1_m3_result-s%:
	# Dev
	# Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00242-t1-v3-full-halo.h5_labels_train.H > t0d.H
	# echo "d1=1.0 d2=1.0" >> t0d.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m3_v1/eegg_train1_m3_v1.h5_labels_2d_dev_file0.H > t1d.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m3_v1/eegg_train1_m3_v1.h5_y_pred_2d_dev_file0.H > t2d.H
	Cat axis=3 t1d.H t2d.H | Grey color=j newclip=1 grid=y titles="Labels (dev):Pred (dev)" | Xtpen pximaps=y &

################################## Test 4 ######################################
eegg_train1_m4_v1:
	rm -rf models_2d/eegg_train1_m4_v1
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m4_v1 --model EGNET --n_epochs 300 --lr 2.0e-3 --device cuda:0 --train_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 64 --half 1 --seed 10

eegg_train1_m4_v2:
	rm -rf models_2d/eegg_train1_m4_v2
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m4_v2 --model EGNET --n_epochs 500 --lr 1.0e-3 --device cuda:1 --train_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 160 --half 1 --seed 10

eegg_train1_m4_v3:
	rm -rf models_2d/eegg_train1_m4_v3
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m4_v3 --model EGNET --n_epochs 500 --lr 5e-3 --device cuda:2 --train_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 160 --half 1 --seed 10

eegg_train1_m4_v4:
	rm -rf models_2d/eegg_train1_m4_v4
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m4_v4 --model EGNET --n_epochs 500 --lr 7.5e-3 --device cuda:3 --train_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 160 --half 1 --seed 10

eegg_train1_m4_v5:
	rm -rf models_2d/eegg_train1_m4_v5
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m4_v5 --model EGNET --n_epochs 500 --lr 1e-2 --device cuda:0 --train_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 160 --half 1 --seed 10

eegg_train1_m4_v6:
	rm -rf models_2d/eegg_train1_m4_v6
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m4_v6 --model EGNET --n_epochs 500 --lr 0.2e-2 --device cuda:1 --train_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 160 --half 1 --seed 10 --lr_decay step --step_size 10 --decay_gamma 0.99

eegg_train1_m4_v7:
	rm -rf models_2d/eegg_train1_m4_v7
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m4_v7 --model EGNET --n_epochs 500 --lr 0.3e-2 --device cuda:2 --train_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 160 --half 1 --seed 10 --lr_decay step --step_size 20 --decay_gamma 0.99

eegg_train1_m4_v8:
	rm -rf models_2d/eegg_train1_m4_v8
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m4_v8 --model EGNET --n_epochs 500 --lr 0.4e-2 --device cuda:3 --train_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 160 --half 1 --seed 10 --lr_decay step --step_size 20 --decay_gamma 0.99

eegg_train1_m4_v9:
	rm -rf models_2d/eegg_train1_m4_v9
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m4_v9 --model EGNET --n_epochs 500 --lr 0.4e-2 --device cuda:0 --train_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 160 --half 1 --seed 10 --lr_decay step --step_size 40 --decay_gamma 0.99

eegg_train1_m4_result-s%:
	# Train
	Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00243-t1-v3-full-halo.h5_labels_train.H > t0d.H
	echo "d1=1.0 d2=1.0" >> t0d.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m4_v7/eegg_train1_m4_v7.h5_labels_2d_train_file0.H > t1d.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m4_v7/eegg_train1_m4_v7.h5_y_pred_2d_train_file0.H > t2d.H
	Cat axis=3 t1d.H t2d.H | Grey color=j newclip=1 grid=y titles="Labels (dev):Pred (dev)" | Xtpen pximaps=y &
	# Dev
	Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00242-t1-v3-full-halo.h5_labels_train.H > t0d.H
	echo "d1=1.0 d2=1.0" >> t0d.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m4_v7/eegg_train1_m4_v7.h5_labels_2d_dev_file0.H > t1d.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m4_v7/eegg_train1_m4_v7.h5_y_pred_2d_dev_file0.H > t2d.H
	Cat axis=3 t1d.H t2d.H | Grey color=j newclip=1 grid=y titles="Labels (dev):Pred (dev)" | Xtpen pximaps=y &

################################## Test 5 ######################################
eegg_train1_m5_v1:
	rm -rf models_2d/eegg_train1_m5_v1
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m5_v1 --model EGNET --n_epochs 500 --lr 0.5e-2 --device cuda:0 --train_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --batch_size 160 --half 1 --seed 10

eegg_train1_m5_v2:
	rm -rf models_2d/eegg_train1_m5_v2
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m5_v2 --model EGNET --n_epochs 500 --lr 0.2e-2 --device cuda:1 --train_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --batch_size 160 --half 1 --seed 10 --lr_decay step --step_size 10 --decay_gamma 0.99

eegg_train1_m5_v3:
	rm -rf models_2d/eegg_train1_m5_v3
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m5_v3 --model EGNET --n_epochs 500 --lr 0.3e-2 --device cuda:2 --train_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --batch_size 160 --half 1 --seed 10 --lr_decay step --step_size 20 --decay_gamma 0.99

eegg_train1_m5_v4:
	rm -rf models_2d/eegg_train1_m5_v4
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m5_v4 --model EGNET --n_epochs 500 --lr 0.4e-2 --device cuda:3 --train_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --batch_size 160 --half 1 --seed 10 --lr_decay step --step_size 40 --decay_gamma 0.99

eegg_train1_m5_v5:
	rm -rf models_2d/eegg_train1_m5_v5
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m5_v5 --model EGNET --n_epochs 10 --lr 2.0e-2 --device cuda:0 --train_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --batch_size 64 --half 1 --seed 10

eegg_train1_m5_v6:
	rm -rf models_2d/eegg_train1_m5_v6
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m5_v6 --model EGNET --n_epochs 10 --lr 2.0e-2 --device cuda:0 --train_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --batch_size 64 --half 0 --seed 10

eegg_train1_m5_result-s%:
	# Train
	Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00243-t1-v4-patch-halo.h5_labels_train.H > t0d.H
	echo "d1=1.0 d2=1.0" >> t0d.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m5_v4/eegg_train1_m5_v4.h5_labels_2d_train_file0.H > t1d.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m5_v4/eegg_train1_m5_v4.h5_y_pred_2d_train_file0.H > t2d.H
	Cat axis=3 t1d.H t2d.H | Grey color=j newclip=1 grid=y titles="Labels:Pred" gainpanel=a bclip=0 eclip=400 | Xtpen pximaps=y &
	# Dev
	Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00242-t1-v3-full-halo.h5_labels_train.H > t0d.H
	echo "d1=1.0 d2=1.0" >> t0d.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m5_v4/eegg_train1_m5_v4.h5_labels_2d_dev_file0.H > t1d.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m5_v4/eegg_train1_m5_v4.h5_y_pred_2d_dev_file0.H > t2d.H
	Cat axis=3 t1d.H t2d.H | Grey color=j newclip=1 grid=y titles="Labels (dev):Pred (dev)" gainpanel=a bclip=0 eclip=400 | Xtpen pximaps=y &

################################## Test 6 ######################################
eegg_train1_m6_v1:
	rm -rf models_2d/eegg_train1_m6_v1
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m6_v1 --model EGNET --n_epochs 400 --lr 0.5e-2 --device cuda:0 --train_file_list par/train_2d_file.txt --dev_file_list par/dev_2d_file.txt --batch_size 32 --half 1 --seed 10

eegg_train1_m6_v2:
	rm -rf models_2d/eegg_train1_m6_v2
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m6_v2 --model EGNET --n_epochs 400 --lr 0.5e-2 --device cuda:1 --train_file_list par/train_2d_file.txt --dev_file_list par/dev_2d_file.txt --batch_size 32 --half 0 --seed 10

eegg_train1_m6_v3:
	rm -rf models_2d/eegg_train1_m6_v3
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m6_v3 --model EGNET --n_epochs 500 --lr 0.1e-2 --device cuda:0 --train_file_list par/train_2d_file.txt --dev_file_list par/dev_2d_file.txt --batch_size 64 --half 1 --seed 10  --lr_decay step --step_size 20 --decay_gamma 0.99

eegg_train1_m6_v4:
	rm -rf models_2d/eegg_train1_m6_v4
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m6_v4 --model EGNET --n_epochs 500 --lr 0.25e-2 --device cuda:1 --train_file_list par/train_2d_file.txt --dev_file_list par/dev_2d_file.txt --batch_size 128 --half 1 --seed 10  --lr_decay step --step_size 50 --decay_gamma 0.95

eegg_train1_m6_v5:
	rm -rf models_2d/eegg_train1_m6_v5
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m6_v5 --model EGNET --n_epochs 500 --lr 0.5e-2 --device cuda:2 --train_file_list par/train_2d_file.txt --dev_file_list par/dev_2d_file.txt --batch_size 128 --half 1 --seed 10  --lr_decay step --step_size 10 --decay_gamma 0.99

eegg_train1_m6_v6:
	rm -rf models_2d/eegg_train1_m6_v6
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m6_v6 --model EGNET --n_epochs 500 --lr 1.0e-2 --device cuda:3 --train_file_list par/train_2d_file.txt --dev_file_list par/dev_2d_file.txt --batch_size 64 --half 1 --seed 10  --lr_decay step --step_size 50 --decay_gamma 0.99

eegg_train1_m6_result-s%:
	# Train
	# Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00243-t1-v3-full-halo.h5_labels_train.H > t0d-f0.H
	# echo "d1=1.0 d2=1.0" >> t0d-f0.H
	# Window3d n3=1 f3=$* < models_2d/eegg_train1_m6_v1/eegg_train1_m6_v1.h5_labels_2d_train_file0.H > t1d-f0.H
	# Window3d n3=1 f3=$* < models_2d/eegg_train1_m6_v1/eegg_train1_m6_v1.h5_y_pred_2d_train_file0.H > t2d-f0.H
	# Cat axis=3 t0d-f0.H t1d-f0.H t2d-f0.H | Grey color=j newclip=1 grid=y titles="Rapid:Labels (dev):Pred (dev)" | Xtpen pximaps=y &
	# Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00242-t1-v3-full-halo.h5_labels_train.H > t0d-f1.H
	# echo "d1=1.0 d2=1.0" >> t0d-f1.H
	# Window3d n3=1 f3=$* < models_2d/eegg_train1_m6_v1/eegg_train1_m6_v1.h5_labels_2d_train_file1.H > t1d-f1.H
	# Window3d n3=1 f3=$* < models_2d/eegg_train1_m6_v1/eegg_train1_m6_v1.h5_y_pred_2d_train_file1.H > t2d-f1.H
	# Cat axis=3 t0d-f1.H t1d-f1.H t2d-f1.H | Grey color=j newclip=1 grid=y titles="Rapid:Labels (dev):Pred (dev)" | Xtpen pximaps=y &
	# Dev
	Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00243-t1-v1-full-halo.h5_labels_train.H > t0t-f0.H.H
	echo "d1=1.0 d2=1.0" >> t0t-f0.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m6_v3/eegg_train1_m6_v3.h5_labels_2d_dev_file0.H > t1t-f0.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m6_v3/eegg_train1_m6_v3.h5_y_pred_2d_dev_file0.H > t2t-f0.H
	Cat axis=3 t0t-f0.H t1t-f0.H t2t-f0.H | Grey color=j newclip=1 grid=y titles="Rapid:Labels (dev):Pred (dev)" | Xtpen pximaps=y &
	Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00242-t1-v1-full-halo.h5_labels_train.H > t0t-f1.H.H
	echo "d1=1.0 d2=1.0" >> t0t-f1.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m6_v3/eegg_train1_m6_v3.h5_labels_2d_dev_file1.H > t1t-f1.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m6_v3/eegg_train1_m6_v3.h5_y_pred_2d_dev_file1.H > t2t-f1.H
	Cat axis=3 t0t-f1.H t1t-f1.H t2t-f1.H | Grey color=j newclip=1 grid=y titles="Rapid:Labels (dev):Pred (dev)" | Xtpen pximaps=y &

################################## Test 7 ######################################
eegg_train1_m7_v1:
	rm -rf models_2d/eegg_train1_m7_v1
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m7_v1 --model EGNET --n_epochs 500 --lr 0.5e-2 --device cuda:0 --train_file dat_2d/eegg-dat-S00233-t1-v1-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --batch_size 64 --half 1 --seed 10

eegg_train1_m7_v2:
	rm -rf models_2d/eegg_train1_m7_v2
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m7_v2 --model EGNET --n_epochs 500 --lr 0.25e-2 --device cuda:1 --train_file dat_2d/eegg-dat-S00233-t1-v1-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --batch_size 64 --half 1 --seed 10

eegg_train1_m7_v3:
	rm -rf models_2d/eegg_train1_m7_v3
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m7_v3 --model EGNET --n_epochs 500 --lr 0.1e-2 --device cuda:2 --train_file dat_2d/eegg-dat-S00233-t1-v1-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --batch_size 64 --half 1 --seed 10

eegg_train1_m7_v4:
	rm -rf models_2d/eegg_train1_m7_v4
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m7_v4 --model EGNET --n_epochs 500 --lr 0.75e-2 --device cuda:3 --train_file dat_2d/eegg-dat-S00233-t1-v1-patch-halo.h5 --dev_file dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 --batch_size 64 --half 1 --seed 10

eegg_train1_m7_result-s%:
	# Train
	Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00233-t1-v1-full-halo.h5_labels_train.H > t0t.H
	echo "d1=1.0 d2=1.0" >> t0t.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m7_v2/eegg_train1_m7_v2.h5_labels_2d_train_file0.H > t1t.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m7_v2/eegg_train1_m7_v2.h5_y_pred_2d_train_file0.H > t2t.H
	Cat axis=3 t0t.H t1t.H t2t.H | Grey color=j newclip=1 grid=y titles="Rapid:Labels:Pred" gainpanel=a bclip=0 eclip=400 | Xtpen pximaps=y &
	# Dev
	Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00243-t1-v3-full-halo.h5_labels_train.H > t0d.H
	echo "d1=1.0 d2=1.0" >> t0d.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m7_v2/eegg_train1_m7_v2.h5_labels_2d_dev_file0.H > t1d.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m7_v2/eegg_train1_m7_v2.h5_y_pred_2d_dev_file0.H > t2d.H
	Cat axis=3 t0d.H t1d.H t2d.H | Grey color=j newclip=1 grid=y titles="Rapid:Labels (dev):Pred (dev)" gainpanel=a bclip=0 eclip=400 | Xtpen pximaps=y &

eegg_train1_m7_stats:
	# Data
	Histogram dinterv=10 min=0 max=150 < dat_2d/eegg-dat-S00233-t1-v1-patch-halo.h5_data_train.H | Scale > t1.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5_data_train.H | Scale > t2.H
	Cat axis=2 t1.H t2.H | Scale | Graph grid=y min1=0.0 max1=150 min2=0.0 max2=1.0 legend=y curvelabel="data train:data dev" legendloc=tr | Xtpen &
	# Labels
	Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S00233-t1-v1-patch-halo.h5_labels_train.H | Scale > t3.H
	Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5_labels_train.H | Scale > t4.H
	Cat axis=2 t3.H t4.H | Scale | Graph grid=y min1=0.0 max1=150 min2=0.0 max2=1.0 legend=y curvelabel="labels train:labels dev" legendloc=tr | Xtpen &

################################## Test 8 ######################################
eegg_train1_m8_v1:
	rm -rf models_2d/eegg_train1_m8_v1
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v1 --model EGNET --n_epochs 500 --lr 0.5e-2 --device cuda:0 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 120 --half 1 --seed 10

eegg_train1_m8_v2:
	rm -rf models_2d/eegg_train1_m8_v2
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v2 --model EGNET --n_epochs 500 --lr 0.25e-2 --device cuda:1 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 120 --half 1 --seed 10

eegg_train1_m8_v3:
	rm -rf models_2d/eegg_train1_m8_v3
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v3 --model EGNET --n_epochs 500 --lr 0.1e-2 --device cuda:2 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 120 --half 1 --seed 10

eegg_train1_m8_v4:
	rm -rf models_2d/eegg_train1_m8_v4
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v4 --model EGNET --n_epochs 500 --lr 0.1e-2 --device cuda:3 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 120 --half 1 --seed 10 --lr_decay step --step_size 20 --decay_gamma 0.99

eegg_train1_m8_v5:
	rm -rf models_2d/eegg_train1_m8_v5
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v5 --model EGNET --n_epochs 500 --lr 0.75e-2 --device cuda:0 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 64 --half 1 --seed 10 --lr_decay step --step_size 10 --decay_gamma 0.99

eegg_train1_m8_v6:
	rm -rf models_2d/eegg_train1_m8_v6
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v6 --model EGNET --n_epochs 500 --lr 1.0e-2 --device cuda:1 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 64 --half 1 --seed 10 --lr_decay step --step_size 10 --decay_gamma 0.99

eegg_train1_m8_v7:
	rm -rf models_2d/eegg_train1_m8_v7
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v7 --model EGNET --n_epochs 500 --lr 0.75e-3 --device cuda:2 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 64 --half 1 --seed 10

eegg_train1_m8_v8:
	rm -rf models_2d/eegg_train1_m8_v8
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v8 --model EGNET --n_epochs 500 --lr 0.5e-3 --device cuda:3 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 64 --half 1 --seed 10

eegg_train1_m8_v9:
	rm -rf models_2d/eegg_train1_m8_v9
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v9 --model EGNET --n_epochs 500 --lr 0.22e-2 --device cuda:0 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 120 --half 1 --seed 10 --lr_decay step --step_size 20 --decay_gamma 0.995

eegg_train1_m8_v10:
	rm -rf models_2d/eegg_train1_m8_v10
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v10 --model EGNET --n_epochs 500 --lr 0.21e-2 --device cuda:1 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 120 --half 1 --seed 10 --lr_decay step --step_size 40 --decay_gamma 0.995

eegg_train1_m8_v11:
	rm -rf models_2d/eegg_train1_m8_v11
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v11 --model EGNET --n_epochs 500 --lr 0.18e-2 --device cuda:2 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 120 --half 1 --seed 10

eegg_train1_m8_v12:
	rm -rf models_2d/eegg_train1_m8_v12
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v12 --model EGNET --n_epochs 500 --lr 0.15e-2 --device cuda:3 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 120 --half 1 --seed 10

eegg_train1_m8_v13:
	rm -rf models_2d/eegg_train1_m8_v13
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v13 --model EGNET --n_epochs 500 --lr 0.15e-2 --device cuda:0 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 64 --half 1 --seed 10 --seed 10 --lr_decay step --step_size 40 --decay_gamma 0.99

eegg_train1_m8_v14:
	rm -rf models_2d/eegg_train1_m8_v14
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v14 --model EGNET --n_epochs 500 --lr 0.14e-2 --device cuda:1 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 64 --half 1 --seed 10 --seed 10 --lr_decay step --step_size 40 --decay_gamma 0.99

eegg_train1_m8_v15:
	rm -rf models_2d/eegg_train1_m8_v15
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v15 --model EGNET --n_epochs 500 --lr 0.12e-2 --device cuda:2 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 64 --half 1 --seed 10 --seed 10 --lr_decay step --step_size 40 --decay_gamma 0.995

eegg_train1_m8_v16:
	rm -rf models_2d/eegg_train1_m8_v16
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v16 --model EGNET --n_epochs 500 --lr 0.11e-2 --device cuda:3 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 64 --half 1 --seed 10 --seed 10 --lr_decay step --step_size 40 --decay_gamma 0.995

eegg_train1_m8_v17:
	rm -rf models_2d/eegg_train1_m8_v17
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v17 --model EGNET --n_epochs 1000 --lr 0.5e-3 --device cuda:0 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 64 --half 1 --seed 10

eegg_train1_m8_v18:
	rm -rf models_2d/eegg_train1_m8_v18
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v18 --model EGNET --n_epochs 1000 --lr 0.5e-3 --device cuda:1 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 128 --half 1 --seed 10

eegg_train1_m8_v19:
	rm -rf models_2d/eegg_train1_m8_v19
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v19 --model EGNET --n_epochs 1000 --lr 0.5e-3 --device cuda:2 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 64 --half 0 --seed 10

eegg_train1_m8_v20:
	rm -rf models_2d/eegg_train1_m8_v20
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v20 --model EGNET --n_epochs 1000 --lr 0.5e-3 --device cuda:3 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 64 --half 0 --seed 10

eegg_train1_m8_v21:
	rm -rf models_2d/eegg_train1_m8_v21
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v21 --model EGNET --n_epochs 600 --lr 0.5e-3 --device cuda:0 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 64 --half 1 --seed 10

eegg_train1_m8_v22:
	rm -rf models_2d/eegg_train1_m8_v22
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v22 --model EGNET --n_epochs 1000 --lr 0.7e-3 --device cuda:1 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 128 --half 1 --seed 10

eegg_train1_m8_v23:
	rm -rf models_2d/eegg_train1_m8_v23
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v23 --model EGNET --n_epochs 500 --lr 0.5e-3 --device cuda:2 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 64 --half 0 --seed 10

eegg_train1_m8_v24:
	rm -rf models_2d/eegg_train1_m8_v24
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m8_v24 --model EGNET --n_epochs 500 --lr 0.5e-3 --device cuda:3 --train_file_list par/train1_m8_train.txt --dev_file dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 --batch_size 64 --half 0 --seed 10

eegg_train1_m8_result-s%:
	# Train
	# Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00243-t1-v3-full-halo.h5_labels_train.H > t0d-f0.H
	# echo "d1=1.0 d2=1.0" >> t0d-f0.H
	# Window3d n3=1 f3=$* < models_2d/eegg_train1_m8_v13/eegg_train1_m8_v13.h5_labels_2d_train_file0.H > t1d-f0.H
	# Window3d n3=1 f3=$* < models_2d/eegg_train1_m8_v13/eegg_train1_m8_v13.h5_y_pred_2d_train_file0.H > t2d-f0.H
	# Cat axis=3 t0d-f0.H t1d-f0.H t2d-f0.H | Grey color=j newclip=1 grid=y titles="Rapid:Labels (dev):Pred (dev)" | Xtpen pximaps=y &
	# Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00233-t1-v1-full-halo.h5_labels_train.H > t0d-f1.H
	# echo "d1=1.0 d2=1.0" >> t0d-f1.H
	# Window3d n3=1 f3=$* < models_2d/eegg_train1_m8_v13/eegg_train1_m8_v13.h5_labels_2d_train_file1.H > t1d-f1.H
	# Window3d n3=1 f3=$* < models_2d/eegg_train1_m8_v13/eegg_train1_m8_v13.h5_y_pred_2d_train_file1.H > t2d-f1.H
	# Cat axis=3 t0d-f1.H t1d-f1.H t2d-f1.H | Grey color=j newclip=1 grid=y titles="Rapid:Labels (dev):Pred (dev)" | Xtpen pximaps=y &
	# Dev
	Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00242-t1-v3-full-halo.h5_labels_train.H > t0t-f0.H
	echo "d1=1.0 d2=1.0" >> t0t-f0.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m8_v23/eegg_train1_m8_v23.h5_labels_2d_dev_file0.H > t1t-f0.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m8_v23/eegg_train1_m8_v23.h5_y_pred_2d_dev_file0.H > t2t-f0.H
	Cat axis=3 t0t-f0.H t1t-f0.H t2t-f0.H | Grey color=j newclip=1 grid=y titles="Rapid:Labels (dev):Pred (dev)" | Xtpen pximaps=y &

################################## Test 9 ######################################
eegg_train1_m9_v1:
	rm -rf models_2d/eegg_train1_m9_v1
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m9_v1 --model EGNET --n_epochs 500 --lr 0.5e-2 --device cuda:0 --train_file_list par/train1_m9_train.txt --dev_file_list par/train1_m9_dev.txt --batch_size 120 --half 1 --seed 10

eegg_train1_m9_v2:
	rm -rf models_2d/eegg_train1_m9_v2
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m9_v2 --model EGNET --n_epochs 500 --lr 0.25e-2 --device cuda:1 --train_file_list par/train1_m9_train.txt --dev_file_list par/train1_m9_dev.txt --batch_size 120 --half 1 --seed 10

eegg_train1_m9_v3:
	rm -rf models_2d/eegg_train1_m9_v3
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m9_v3 --model EGNET --n_epochs 500 --lr 0.1e-2 --device cuda:2 --train_file_list par/train1_m9_train.txt --dev_file_list par/train1_m9_dev.txt --batch_size 120 --half 1 --seed 10

eegg_train1_m9_v4:
	rm -rf models_2d/eegg_train1_m9_v4
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m9_v4 --model EGNET --n_epochs 500 --lr 0.6e-2 --device cuda:3 --train_file_list par/train1_m9_train.txt --dev_file_list par/train1_m9_dev.txt --batch_size 120 --half 1 --seed 10

eegg_train1_m9_v5:
	rm -rf models_2d/eegg_train1_m9_v5
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m9_v5 --model EGNET --n_epochs 500 --lr 0.75e-3 --device cuda:0 --train_file_list par/train1_m9_train.txt --dev_file_list par/train1_m9_dev.txt --batch_size 120 --half 1 --seed 10

eegg_train1_m9_v6:
	rm -rf models_2d/eegg_train1_m9_v6
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m9_v6 --model EGNET --n_epochs 500 --lr 0.5e-3 --device cuda:1 --train_file_list par/train1_m9_train.txt --dev_file_list par/train1_m9_dev.txt --batch_size 120 --half 1 --seed 10

eegg_train1_m9_v7:
	rm -rf models_2d/eegg_train1_m9_v7
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m9_v7 --model EGNET --n_epochs 500 --lr 0.25e-3 --device cuda:2 --train_file_list par/train1_m9_train.txt --dev_file_list par/train1_m9_dev.txt --batch_size 120 --half 1 --seed 10  --lr_decay step --step_size 20 --decay_gamma 0.995

eegg_train1_m9_v8:
	rm -rf models_2d/eegg_train1_m9_v8
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m9_v8 --model EGNET --n_epochs 500 --lr 0.1e-3 --device cuda:3 --train_file_list par/train1_m9_train.txt --dev_file_list par/train1_m9_dev.txt --batch_size 120 --half 1 --seed 10  --lr_decay step --step_size 10 --decay_gamma 0.995

eegg_train1_m9_v9:
	rm -rf models_2d/eegg_train1_m9_v9
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m9_v9 --model EGNET --n_epochs 500 --lr 0.75e-3 --device cuda:0 --train_file_list par/train1_m9_train.txt --dev_file_list par/train1_m9_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay step --step_size 20 --decay_gamma 0.995

eegg_train1_m9_v10:
	rm -rf models_2d/eegg_train1_m9_v10
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m9_v10 --model EGNET --n_epochs 500 --lr 0.75e-3 --device cuda:1 --train_file_list par/train1_m9_train.txt --dev_file_list par/train1_m9_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay step --step_size 40 --decay_gamma 0.995

eegg_train1_m9_v11:
	rm -rf models_2d/eegg_train1_m9_v11
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m9_v11 --model EGNET --n_epochs 500 --lr 0.75e-3 --device cuda:2 --train_file_list par/train1_m9_train.txt --dev_file_list par/train1_m9_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay step --step_size 50 --decay_gamma 0.99

eegg_train1_m9_v12:
	rm -rf models_2d/eegg_train1_m9_v12
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m9_v12 --model EGNET --n_epochs 500 --lr 0.75e-3 --device cuda:3 --train_file_list par/train1_m9_train.txt --dev_file_list par/train1_m9_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay step --step_size 50 --decay_gamma 0.9

eegg_train1_m9_v13:
	rm -rf models_2d/eegg_train1_m9_v13
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m9_v13 --model EGNET --n_epochs 1000 --lr 0.75e-3 --device cuda:0 --train_file_list par/train1_m9_train.txt --dev_file_list par/train1_m9_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay step --step_size 20 --decay_gamma 0.995

eegg_train1_m9_v14:
	rm -rf models_2d/eegg_train1_m9_v14
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m9_v14 --model EGNET --n_epochs 1000 --lr 0.75e-3 --device cuda:1 --train_file_list par/train1_m9_train.txt --dev_file_list par/train1_m9_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay step --step_size 40 --decay_gamma 0.995

eegg_train1_m9_v15:
	rm -rf models_2d/eegg_train1_m9_v15
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m9_v15 --model EGNET --n_epochs 1000 --lr 0.75e-3 --device cuda:2 --train_file_list par/train1_m9_train.txt --dev_file_list par/train1_m9_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay step --step_size 50 --decay_gamma 0.99

eegg_train1_m9_v16:
	rm -rf models_2d/eegg_train1_m9_v16
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m9_v16 --model EGNET --n_epochs 1000 --lr 0.75e-3 --device cuda:3 --train_file_list par/train1_m9_train.txt --dev_file_list par/train1_m9_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay step --step_size 50 --decay_gamma 0.9

eegg_train1_m9_result-s%:
	# Train
	Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00239-t1-full-halo.h5_labels_train.H > t0d-f0.H
	echo "d1=1.0 d2=1.0" >> t0d-f0.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m9_v5/eegg_train1_m9_v5.h5_labels_2d_dev_file0.H > t1d-f0.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m9_v5/eegg_train1_m9_v5.h5_y_pred_2d_dev_file0.H > t2d-f0.H
	Cat axis=3 t0d-f0.H t1d-f0.H t2d-f0.H | Grey color=j newclip=1 grid=y titles="Rapid:Labels (dev):Pred (dev)" | Xtpen pximaps=y &

################################## Test 10 #####################################
eegg_train1_m10_v1:
	rm -rf models_2d/eegg_train1_m10_v1
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v1 --model EGNET --n_epochs 500 --lr 1e-3 --device cuda:0 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10

eegg_train1_m10_v2:
	rm -rf models_2d/eegg_train1_m10_v2
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v2 --model EGNET --n_epochs 500 --lr 1.25e-3 --device cuda:1 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10

eegg_train1_m10_v3:
	rm -rf models_2d/eegg_train1_m10_v3
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v3 --model EGNET --n_epochs 500 --lr 1.4e-3 --device cuda:2 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10

eegg_train1_m10_v4:
	rm -rf models_2d/eegg_train1_m10_v4
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v4 --model EGNET --n_epochs 500 --lr 1.75e-3 --device cuda:3 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10

eegg_train1_m10_v5:
	rm -rf models_2d/eegg_train1_m10_v5
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v5 --model EGNET --n_epochs 500 --lr 1e-3 --device cuda:0 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay step --step_size 50 --decay_gamma 0.9

eegg_train1_m10_v6:
	rm -rf models_2d/eegg_train1_m10_v6
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v6 --model EGNET --n_epochs 500 --lr 1e-3 --device cuda:1 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay step --step_size 10 --decay_gamma 0.95

eegg_train1_m10_v7:
	rm -rf models_2d/eegg_train1_m10_v7
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v7 --model EGNET --n_epochs 500 --lr 1e-3 --device cuda:2 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay step --step_size 20 --decay_gamma 0.95

eegg_train1_m10_v8:
	rm -rf models_2d/eegg_train1_m10_v8
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v8 --model EGNET --n_epochs 500 --lr 1e-3 --device cuda:3 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay step --step_size 10 --decay_gamma 0.99

eegg_train1_m10_v9:
	rm -rf models_2d/eegg_train1_m10_v9
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v9 --model EGNET --n_epochs 500 --lr 1e-3 --device cuda:0 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay step --step_size 10 --decay_gamma 0.9

eegg_train1_m10_v10:
	rm -rf models_2d/eegg_train1_m10_v10
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v10 --model EGNET --n_epochs 500 --lr 1e-3 --device cuda:1 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay decay --decay_rate 0.001

eegg_train1_m10_v11:
	rm -rf models_2d/eegg_train1_m10_v11
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v11 --model EGNET --n_epochs 500 --lr 1e-3 --device cuda:2 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay step --step_size 5 --decay_gamma 0.9

eegg_train1_m10_v12:
	rm -rf models_2d/eegg_train1_m10_v12
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v12 --model EGNET --n_epochs 500 --lr 1e-3 --device cuda:3 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay step --step_size 10 --decay_gamma 0.9

eegg_train1_m10_v13:
	rm -rf models_2d/eegg_train1_m10_v13
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v13 --model EGNET --n_epochs 500 --lr 1e-3 --device cuda:0 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay decay --decay_rate 0.002

eegg_train1_m10_v14:
	rm -rf models_2d/eegg_train1_m10_v14
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v14 --model EGNET --n_epochs 500 --lr 1e-3 --device cuda:1 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay decay --decay_rate 0.004

eegg_train1_m10_v15:
	rm -rf models_2d/eegg_train1_m10_v15
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v15 --model EGNET --n_epochs 500 --lr 1e-3 --device cuda:2 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay decay --decay_rate 0.006

eegg_train1_m10_v16:
	rm -rf models_2d/eegg_train1_m10_v16
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v16 --model EGNET --n_epochs 500 --lr 1e-3 --device cuda:3 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay decay --decay_rate 0.009

eegg_train1_m10_v17:
	rm -rf models_2d/eegg_train1_m10_v17
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v17 --model EGNET --n_epochs 1000 --lr 1e-3 --device cuda:0 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay decay --decay_rate 0.012

eegg_train1_m10_v18:
	rm -rf models_2d/eegg_train1_m10_v18
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v18 --model EGNET --n_epochs 1000 --lr 1e-3 --device cuda:1 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay decay --decay_rate 0.015

eegg_train1_m10_v19:
	rm -rf models_2d/eegg_train1_m10_v19
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v19 --model EGNET --n_epochs 500 --lr 1e-3 --device cuda:2 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay decay --decay_rate 0.02

eegg_train1_m10_v20:
	rm -rf models_2d/eegg_train1_m10_v20
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v20 --model EGNET --n_epochs 500 --lr 1e-3 --device cuda:3 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10 --lr_decay step --step_size 10 --decay_gamma 0.8

eegg_train1_m10_v21:
	rm -rf models_2d/eegg_train1_m10_v21
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v21 --model EGNET --n_epochs 500 --lr 2e-3 --device cuda:0 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 40 --half 1 --seed 10 --loss huber

eegg_train1_m10_v22:
	rm -rf models_2d/eegg_train1_m10_v22
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v22 --model EGNET --n_epochs 500 --lr 1e-4 --device cuda:1 --train_file_list par/train1_m10_train.txt --dev_file_list par/train1_m10_dev.txt --batch_size 120 --half 1 --seed 10 --loss huber

eegg_train1_m10_v23:
	rm -rf models_2d/eegg_train1_m10_v23
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v23 --model EGNET --n_epochs 1000 --lr 1e-3 --device cuda:0 --train_file_list par/train1_m10_train_clip.txt --dev_file_list par/train1_m10_dev_clip.txt --batch_size 120 --half 1 --seed 10

eegg_train1_m10_v24:
	rm -rf models_2d/eegg_train1_m10_v24
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v24 --model EGNET --n_epochs 1000 --lr 0.5e-3 --device cuda:1 --train_file_list par/train1_m10_train_clip.txt --dev_file_list par/train1_m10_dev_clip.txt --batch_size 120 --half 1 --seed 10

eegg_train1_m10_v25:
	rm -rf models_2d/eegg_train1_m10_v25
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v25 --model EGNET --n_epochs 1000 --lr 1.2e-3 --device cuda:2 --train_file_list par/train1_m10_train_clip.txt --dev_file_list par/train1_m10_dev_clip.txt --batch_size 120 --half 1 --seed 10 --lr_decay decay --decay_rate 0.015

eegg_train1_m10_v26:
	rm -rf models_2d/eegg_train1_m10_v26
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m10_v26 --model EGNET --n_epochs 1000 --lr 1.5e-3 --device cuda:3 --train_file_list par/train1_m10_train_clip.txt --dev_file_list par/train1_m10_dev_clip.txt --batch_size 120 --half 1 --seed 10 --lr_decay decay --decay_rate 0.015

eegg_train1_m10_result_train_s%:
	# Train
	Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00243-t1-v3-full-halo.h5_labels_train.H > t0d-f0.H
	# Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00233-t1-v1-full-halo_clip.h5_labels_train.H > t0d-f0.H
	echo "d1=1.0 d2=1.0" >> t0d-f0.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m10_v25/eegg_train1_m10_v25.h5_labels_2d_train_file0.H > t1d-f0.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m10_v25/eegg_train1_m10_v25.h5_y_pred_2d_train_file0.H > t2d-f0.H
	Cat axis=3 t0d-f0.H t1d-f0.H t2d-f0.H | Grey color=j newclip=1 grid=y titles="Rapid:Labels (dev):Pred (dev)" gainpanel=a bclip=0 eclip=200 | Xtpen pximaps=y &

eegg_train1_m10_result_dev_s%:
	# Train
	Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00239-t1-full-halo.h5_labels_train.H > t0d-f0.H
	# Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00233-t1-v1-full-halo_clip.h5_labels_train.H > t0d-f0.H
	echo "d1=1.0 d2=1.0" >> t0d-f0.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m10_v25/eegg_train1_m10_v25.h5_labels_2d_dev_file0.H > t1d-f0.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m10_v25/eegg_train1_m10_v25.h5_y_pred_2d_dev_file0.H > t2d-f0.H
	Cat axis=3 t1d-f0.H t2d-f0.H | Grey color=j newclip=1 grid=y titles="True:Pred" gainpanel=a bclip=0 eclip=200 | Xtpen pximaps=y &

################################## Test 11 #####################################
eegg_train1_m11_v1:
	rm -rf models_2d/eegg_train1_m11_v1
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m11_v1 --model EGNET --n_epochs 200 --lr 1.2e-3 --device cuda:0 --train_file_list par/train1_m11_train_clip.txt --dev_file_list par/train1_m11_dev_clip.txt --batch_size 120 --half 1 --seed 10 --lr_decay decay --decay_rate 0.015

eegg_train1_m11_v2:
	rm -rf models_2d/eegg_train1_m11_v2
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m11_v2 --model EGNET --n_epochs 200 --lr 1.0e-3 --device cuda:1 --train_file_list par/train1_m11_train_clip.txt --dev_file_list par/train1_m11_dev_clip.txt --batch_size 120 --half 1 --seed 10 --lr_decay decay --decay_rate 0.02

eegg_train1_m11_v3:
	rm -rf models_2d/eegg_train1_m11_v3
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m11_v3 --model EGNET --n_epochs 200 --lr 1.0e-3 --device cuda:2 --train_file_list par/train1_m11_train_clip.txt --dev_file_list par/train1_m11_dev_clip.txt --batch_size 120 --half 1 --seed 10 --lr_decay decay --decay_rate 0.03

eegg_train1_m11_v4:
	rm -rf models_2d/eegg_train1_m11_v4
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m11_v4 --model EGNET --n_epochs 200 --lr 1.0e-3 --device cuda:3 --train_file_list par/train1_m11_train_clip.txt --dev_file_list par/train1_m11_dev_clip.txt --batch_size 120 --half 1 --seed 10

eegg_train1_m11_v5:
	rm -rf models_2d/eegg_train1_m11_v5
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m11_v5 --model EGNET --n_epochs 500 --lr 1.0e-3 --device cuda:2 --train_file_list par/train1_m11_train_clip.txt --dev_file_list par/train1_m11_dev_clip.txt --batch_size 120 --half 1 --seed 10 --lr_decay decay --decay_rate 0.03

eegg_train1_m11_v6:
	rm -rf models_2d/eegg_train1_m11_v6
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m11_v6 --model EGNET --n_epochs 600 --lr 1.0e-3 --device cuda:3 --train_file_list par/train1_m11_train_clip.txt --dev_file_list par/train1_m11_dev_clip.txt --batch_size 120 --half 1 --seed 10

eegg_train1_m11_v7:
	rm -rf models_2d/eegg_train1_m11_v7
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m11_v7 --model EGNET --n_epochs 600 --lr 1.0e-3 --device cuda:1 --train_file_list par/train1_m11_train_clip.txt --dev_file_list par/train1_m11_dev_clip.txt --batch_size 120 --half 1 --seed 10 --lr_decay decay --decay_rate 0.02

eegg_train1_m11_v8:
	rm -rf models_2d/eegg_train1_m11_v8
	python3.6 ./python/CTP_main_2d.py train eegg_train1_m11_v8 --model EGNET --n_epochs 600 --lr 1.2e-3 --device cuda:0 --train_file_list par/train1_m11_train_clip.txt --dev_file_list par/train1_m11_dev_clip.txt --batch_size 120 --half 1 --seed 10 --lr_decay decay --decay_rate 0.015


eegg_train1_m11_result_train_s%:
	# Train
	# Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00243-t1-v3-full-halo.h5_labels_train.H > t0d-f0.H
	# Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00233-t1-v1-full-halo_clip.h5_labels_train.H > t0d-f0.H
	# Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00242-t1-v3-full-halo.h5_labels_train.H > t0d-f0.H
	Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00286-t1-full-halo.h5_labels_train.H > t0d-f0.H
	echo "d1=1.0 d2=1.0" >> t0d-f0.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m11_v2/eegg_train1_m11_v2.h5_labels_2d_train_file7.H > t1d-f0.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m11_v2/eegg_train1_m11_v2.h5_y_pred_2d_train_file7.H > t2d-f0.H
	Cat axis=3 t1d-f0.H t2d-f0.H | Grey color=j newclip=1 grid=y titles="True:Pred" gainpanel=a bclip=0 eclip=200 | Xtpen pximaps=y &

eegg_train1_m10_result_dev_s%:
	# Train
	# Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00239-t1-full-halo.h5_labels_train.H > t0d-f0.H
	Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00297-t1-full-halo.h5_labels_train.H > t0d-f0.H
	echo "d1=1.0 d2=1.0" >> t0d-f0.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m11_v6/eegg_train1_m11_v6.h5_labels_2d_dev_file1.H > t1d-f0.H
	Window3d n3=1 f3=$* < models_2d/eegg_train1_m11_v6/eegg_train1_m11_v6.h5_y_pred_2d_dev_file1.H > t2d-f0.H
	Cat axis=3 t1d-f0.H t2d-f0.H | Grey color=j newclip=1 grid=y titles="True:Pred" gainpanel=a bclip=0 eclip=200 | Xtpen pximaps=y &
