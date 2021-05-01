################################################################################
################################## Data ########################################
################################################################################

################################## S00243 ######################################
# 1 slice
dat_2d/eegg-dat-S00243-t1-v1-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 40 --skip 0 1 2 4 5 6 7 8 9 --halo 5

dat_2d/eegg-dat-S00243-t1-v1-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 162 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 40 --skip 0 1 2 4 5 6 7 8 9

dat_2d/eegg-dat-S00243-t1-v1-total-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 256 --y_patch_size 256 --nx_patch 1 --ny_patch 1 --ox_patch 0 --oy_patch 0 --skip 0 1 2 4 5 6 7 8 9

# 1 slice
dat_2d/eegg-dat-S00243-t1-v2-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 40 --skip 0 2 3 4 5 6 7 8 9 --halo 5

dat_2d/eegg-dat-S00243-t1-v2-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 162 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 40 --skip 0 2 3 4 5 6 7 8 9

dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 40 --halo 5

dat_2d/eegg-dat-S00243-t1-v3-patch-halo_clip.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 40 --halo 5 --clip 200

dat_2d/eegg-dat-S00243-t1-v3-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 162 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 40

dat_2d/eegg-dat-S00243-t1-v4-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 40 --halo 5 --skip 3

dat_2d/eegg-dat-S00243-t1-v4-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 162 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 40 --skip 3

makedat1:
	# make dat_2d/eegg-dat-S00243-t1-v1-patch-halo.h5 -B
	# make dat_2d/eegg-dat-S00243-t1-v1-full-halo.h5 -B
	# make dat_2d/eegg-dat-S00243-t1-v2-patch-halo.h5 -B
	# make dat_2d/eegg-dat-S00243-t1-v2-full-halo.h5 -B
	# make dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5 -B
	make dat_2d/eegg-dat-S00243-t1-v4-patch-halo.h5 -B
	make dat_2d/eegg-dat-S00243-t1-v4-full-halo.h5 -B

################################## S00242 ######################################
dat_2d/eegg-dat-S00242-t1-v1-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00242/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00242/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 20 --oy_patch 20 --halo 5 --skip 0 1 2 4 5 6 7 8 9 --halo 5

dat_2d/eegg-dat-S00242-t1-v1-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00242/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00242/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 162 --nx_patch 1 --ny_patch 1 --ox_patch 20 --oy_patch 20	--skip 0 1 2 4 5 6 7 8 9

dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00242/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00242/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 20 --oy_patch 20 --halo 5

dat_2d/eegg-dat-S00242-t1-v3-patch-halo_clip.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00242/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00242/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 20 --oy_patch 20 --halo 5	--clip 200

dat_2d/eegg-dat-S00242-t1-v3-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00242/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00242/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 162 --nx_patch 1 --ny_patch 1 --ox_patch 20 --oy_patch 20

dat_2d/eegg-dat-S00242-t1-v4-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00242/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00242/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 20 --oy_patch 20 --halo 5 --skip 3

dat_2d/eegg-dat-S00242-t1-v4-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00242/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00242/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 162 --nx_patch 1 --ny_patch 1 --ox_patch 20 --oy_patch 20	--skip 3

makedat2:
	make dat_2d/eegg-dat-S00242-t1-v1-patch-halo.h5 -B
	make dat_2d/eegg-dat-S00242-t1-v1-full-halo.h5 -B
	# make dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5 -B
	# make dat_2d/eegg-dat-S00242-t1-v3-full-halo.h5 -B
	# make dat_2d/eegg-dat-S00242-t1-v4-patch-halo.h5 -B
	# make dat_2d/eegg-dat-S00242-t1-v4-full-halo.h5 -B

################################## S00233 ######################################
dat_2d/eegg-dat-S00233-t1-total-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00233/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00233/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 256 --y_patch_size 256 --nx_patch 1 --ny_patch 1 --ox_patch 0 --oy_patch 0

dat_2d/eegg-dat-S00233-t1-v1-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00233/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00233/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 25 --oy_patch 25 --skip 0 1 6 7 8 9 --halo 5

dat_2d/eegg-dat-S00233-t1-v1-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00233/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00233/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 162 --nx_patch 1 --ny_patch 1 --ox_patch 25 --oy_patch 25 --skip 0 1 6 7 8 9

dat_2d/eegg-dat-S00233-t1-v1-patch-halo_clip.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00233/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00233/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 25 --oy_patch 25 --skip 0 1 6 7 8 9 --halo 5 --clip 200

################################## S00239 ######################################
dat_2d/eegg-dat-S00239-t1-total-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00239/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00239/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 256 --y_patch_size 256 --nx_patch 1 --ny_patch 1 --ox_patch 0 --oy_patch 0

dat_2d/eegg-dat-S00239-t1-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00239/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00239/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 216 --nx_patch 1 --ny_patch 1 --ox_patch 35 --oy_patch 20 --skip 7 8 9

dat_2d/eegg-dat-S00239-t1-v1-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00239/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00239/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 4 --ox_patch 35 --oy_patch 20 --halo 5 --skip 7 8 9

dat_2d/eegg-dat-S00239-t1-v1-patch-halo_clip.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00239/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00239/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 4 --ox_patch 35 --oy_patch 20 --halo 5 --skip 7 8 9 --clip 200

################################## S00250 ######################################
dat_2d/eegg-dat-S00250-t1-total-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00250/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00250/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 256 --y_patch_size 256 --nx_patch 1 --ny_patch 1 --ox_patch 0 --oy_patch 0

dat_2d/eegg-dat-S00250-t1-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00250/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00250/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 162 --nx_patch 1 --ny_patch 1 --ox_patch 55 --oy_patch 50 --skip 7 8 9

dat_2d/eegg-dat-S00250-t1-v1-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00250/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00250/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 55 --oy_patch 50 --halo 5 --skip 7 8 9

dat_2d/eegg-dat-S00250-t1-v1-patch-halo_clip.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00250/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00250/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 55 --oy_patch 50 --halo 5 --skip 7 8 9 --clip 200

################################## Stats #######################################
test_clip-s%:
	/opt/SEP/SEP7.0/bin/Clip chop=g clip=200 < dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5_labels_train.H > temp.H
	Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5_labels_train.H > t1.H
	Window3d n3=1 f3=$* < temp.H > t2.H
	Cat axis=3 t1.H t2.H | Grey color=j newclip=1 bclip=0 eclip=400 titles="Original:Clipped" | Xtpen pixmaps=y &

################################## S00254 ######################################
dat_2d/eegg-dat-S00254-t1-total-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00254/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00254/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 256 --y_patch_size 256 --nx_patch 1 --ny_patch 1 --ox_patch 0 --oy_patch 0

dat_2d/eegg-dat-S00254-t1-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00254/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00254/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 216 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 20 --skip 6 7 8 9

dat_2d/eegg-dat-S00254-t1-v1-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00254/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00254/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 4 --ox_patch 40 --oy_patch 20 --skip 6 7 8 9 --halo 5

dat_2d/eegg-dat-S00254-t1-v1-patch-halo_clip.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00254/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00254/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 4 --ox_patch 40 --oy_patch 20 --skip 6 7 8 9 --halo 5 --clip 200

################################## S000271 #####################################
dat_2d/eegg-dat-S000271-t1-total-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00271/Ct\ Head\ Perfusion\ Wcontrast/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00271/Ct\ Head\ Perfusion\ Wcontrast/RAPID\ TMax\ \[s\]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 256 --y_patch_size 256 --nx_patch 1 --ny_patch 1 --ox_patch 0 --oy_patch 0

dat_2d/eegg-dat-S000271-t1-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00271/Ct\ Head\ Perfusion\ Wcontrast/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00271/Ct\ Head\ Perfusion\ Wcontrast/RAPID\ TMax\ \[s\]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 162 --nx_patch 1 --ny_patch 1 --ox_patch 35 --oy_patch 40 --skip 1 3 4 7 8

dat_2d/eegg-dat-S000271-t1-v1-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00271/Ct\ Head\ Perfusion\ Wcontrast/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00271/Ct\ Head\ Perfusion\ Wcontrast/RAPID\ TMax\ \[s\]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 35 --oy_patch 40 --skip 1 3 4 7 8 --halo 5

dat_2d/eegg-dat-S000271-t1-v1-patch-halo_clip.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00271/Ct\ Head\ Perfusion\ Wcontrast/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00271/Ct\ Head\ Perfusion\ Wcontrast/RAPID\ TMax\ \[s\]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 35 --oy_patch 40 --skip 1 3 4 7 8 --halo 5 --clip 200

################################## S00275 ######################################
dat_2d/eegg-dat-S00275-t1-total-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00275/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00275/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 256 --y_patch_size 256 --nx_patch 1 --ny_patch 1 --ox_patch 0 --oy_patch 0

dat_2d/eegg-dat-S00275-t1-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00275/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00275/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 162 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 40 --skip 5 6 7 8 9

dat_2d/eegg-dat-S00275-t1-v1-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00275/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00275/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 40 --skip 5 6 7 8 9 --halo 5

dat_2d/eegg-dat-S00275-t1-v1-patch-halo_clip.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00275/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00275/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 40 --skip 5 6 7 8 9 --halo 5 --clip 200

################################## S00286 ######################################
dat_2d/eegg-dat-S00286-t1-total-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00286/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 3/ raw_data/SS00286/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 333/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 256 --y_patch_size 256 --nx_patch 1 --ny_patch 1 --ox_patch 0 --oy_patch 0

dat_2d/eegg-dat-S00286-t1-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00286/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 3/ raw_data/SS00286/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 333/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 162 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 35 --skip 0 1

dat_2d/eegg-dat-S00286-t1-v1-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00286/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 3/ raw_data/SS00286/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 333/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 35 --skip 0 1 --halo 5

dat_2d/eegg-dat-S00286-t1-v1-patch-halo_clip.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00286/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 3/ raw_data/SS00286/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 333/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 35 --skip 0 1 --halo 5 --clip 200

################################## S00287 ######################################
dat_2d/eegg-dat-S00287-t1-total-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00287/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00287/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 256 --y_patch_size 256 --nx_patch 1 --ny_patch 1 --ox_patch 0 --oy_patch 0

dat_2d/eegg-dat-S00287-t1-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00287/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00287/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 216 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 25 --skip 0 1 2 7 8 9

dat_2d/eegg-dat-S00287-t1-v1-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00287/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00287/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 4 --ox_patch 40 --oy_patch 25 --skip 0 1 2 7 8 9 --halo 5

dat_2d/eegg-dat-S00287-t1-v1-patch-halo_clip.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00287/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00287/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 4 --ox_patch 40 --oy_patch 25 --skip 0 1 2 7 8 9 --halo 5 --clip 200

################################## S00288 ######################################
dat_2d/eegg-dat-S00288-t1-total-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00288/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00288/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 256 --y_patch_size 256 --nx_patch 1 --ny_patch 1 --ox_patch 0 --oy_patch 0

dat_2d/eegg-dat-S00288-t1-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00288/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00288/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 216 --nx_patch 1 --ny_patch 1 --ox_patch 44 --oy_patch 30

dat_2d/eegg-dat-S00288-t1-v1-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00288/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00288/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 4 --ox_patch 44 --oy_patch 30 --halo 5

dat_2d/eegg-dat-S00288-t1-v1-patch-halo_clip.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00288/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00288/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 4 --ox_patch 44 --oy_patch 30 --halo 5 --clip 200

################################## S00289 ######################################
dat_2d/eegg-dat-S00289-t1-total-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00289/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00289/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 256 --y_patch_size 256 --nx_patch 1 --ny_patch 1 --ox_patch 0 --oy_patch 0

dat_2d/eegg-dat-S00289-t1-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00289/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00289/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 216 --nx_patch 1 --ny_patch 1 --ox_patch 30 --oy_patch 15 --skip 6 7 8 9

dat_2d/eegg-dat-S00289-t1-v1-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00289/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00289/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 4 --ox_patch 30 --oy_patch 15 --skip 6 7 8 9 --halo 5

dat_2d/eegg-dat-S00289-t1-v1-patch-halo_clip.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00289/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00289/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 4 --ox_patch 30 --oy_patch 15 --skip 6 7 8 9 --halo 5 --clip 200

################################## S00291 ######################################
dat_2d/eegg-dat-S00291-t1-total-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00291/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00291/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 256 --y_patch_size 256 --nx_patch 1 --ny_patch 1 --ox_patch 0 --oy_patch 0

dat_2d/eegg-dat-S00291-t1-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00291/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00291/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 216 --nx_patch 1 --ny_patch 1 --ox_patch 35 --oy_patch 20 --skip 0 1 9

dat_2d/eegg-dat-S00291-t1-v1-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00291/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00291/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 4 --ox_patch 35 --oy_patch 20 --skip 0 1 9 --halo 5

dat_2d/eegg-dat-S00291-t1-v1-patch-halo_clip.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00291/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00291/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 4 --ox_patch 35 --oy_patch 20 --skip 0 1 9 --halo 5 --clip 200

################################## S00292 ######################################
dat_2d/eegg-dat-S00292-t1-total-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00292/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00292/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 256 --y_patch_size 256 --nx_patch 1 --ny_patch 1 --ox_patch 0 --oy_patch 0

dat_2d/eegg-dat-S00292-t1-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00292/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00292/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 216 --nx_patch 1 --ny_patch 1 --ox_patch 35 --oy_patch 30 --skip 7 8 9

dat_2d/eegg-dat-S00292-t1-v1-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00292/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00292/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 4 --ox_patch 35 --oy_patch 30 --skip 7 8 9 --halo 5

dat_2d/eegg-dat-S00292-t1-v1-patch-halo_clip.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00292/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00292/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 4 --ox_patch 35 --oy_patch 30 --skip 7 8 9 --halo 5 --clip 200

################################## S00293 ######################################
dat_2d/eegg-dat-S00293-t1-total-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00293/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00293/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 256 --y_patch_size 256 --nx_patch 1 --ny_patch 1 --ox_patch 0 --oy_patch 0

dat_2d/eegg-dat-S00293-t1-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00293/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00293/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 216 --nx_patch 1 --ny_patch 1 --ox_patch 30 --oy_patch 20 --skip 0 1 4 9

dat_2d/eegg-dat-S00293-t1-v1-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00293/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00293/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 4 --ox_patch 30 --oy_patch 20 --skip 0 1 4 9 --halo 5

dat_2d/eegg-dat-S00293-t1-v1-patch-halo_clip.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00293/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00293/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 4 --ox_patch 30 --oy_patch 20 --skip 0 1 4 9 --halo 5 --clip 200

################################## S00295 ######################################
dat_2d/eegg-dat-S00295-t1-total-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00295/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00295/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 256 --y_patch_size 256 --nx_patch 1 --ny_patch 1 --ox_patch 0 --oy_patch 0

dat_2d/eegg-dat-S00295-t1-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00295/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00295/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 216 --nx_patch 1 --ny_patch 1 --ox_patch 45 --oy_patch 30 --skip 8 9

dat_2d/eegg-dat-S00295-t1-v1-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00295/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00295/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 4 --ox_patch 45 --oy_patch 30 --skip 8 9 --halo 5

dat_2d/eegg-dat-S00295-t1-v1-patch-halo_clip.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00295/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00295/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 4 --ox_patch 45 --oy_patch 30 --skip 8 9 --halo 5 --clip 200

################################## S00297 ######################################
dat_2d/eegg-dat-S00297-t1-total-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00297/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00297/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 256 --y_patch_size 256 --nx_patch 1 --ny_patch 1 --ox_patch 0 --oy_patch 0

dat_2d/eegg-dat-S00297-t1-full-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00297/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00297/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 162 --y_patch_size 216 --nx_patch 1 --ny_patch 1 --ox_patch 25 --oy_patch 10 --skip 0 6 7

dat_2d/eegg-dat-S00297-t1-v1-patch-halo.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00297/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00297/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 4 --ox_patch 25 --oy_patch 10 --skip 0 6 7 --halo 5

########## ----> 
dat_2d/eegg-dat-S00297-t1-v1-patch-halo_clip.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00297/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00297/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 54 --y_patch_size 54 --nx_patch 3 --ny_patch 4 --ox_patch 25 --oy_patch 10 --skip 0 6 7 --halo 5 --clip 200

################################## Launch ######################################
makeDatLaunch:
	make dat_2d/eegg-dat-S00275-t1-total-halo.h5 -B
	make dat_2d/eegg-dat-S00275-t1-full-halo.h5 -B
	make dat_2d/eegg-dat-S00275-t1-v1-patch-halo.h5 -B
	make dat_2d/eegg-dat-S00286-t1-total-halo.h5 -B
	make dat_2d/eegg-dat-S00286-t1-full-halo.h5 -B
	make dat_2d/eegg-dat-S00286-t1-v1-patch-halo.h5 -B

################################## Statistics ##################################
# Data
eegg_train1_stats_data:
	# Data
	Histogram dinterv=10 min=0 max=150 < dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5_data_train.H | Scale > t1.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/eegg-dat-S00233-t1-v1-patch-halo.h5_data_train.H | Scale > t2.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5_data_train.H | Scale > t3.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/eegg-dat-S00250-t1-v1-patch-halo.h5_data_train.H | Scale > t4.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/eegg-dat-S00239-t1-v1-patch-halo.h5_data_train.H | Scale > t5.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/eegg-dat-S00254-t1-v1-patch-halo.h5_data_train.H | Scale > t6.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/eegg-dat-S000271-t1-v1-patch-halo.h5_data_train.H | Scale > t7.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/eegg-dat-S00275-t1-v1-patch-halo.h5_data_train.H | Scale > t8.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/eegg-dat-S00286-t1-v1-patch-halo.h5_data_train.H | Scale > t9.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/eegg-dat-S00287-t1-v1-patch-halo.h5_data_train.H | Scale > t10.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/eegg-dat-S00288-t1-v1-patch-halo.h5_data_train.H | Scale > t11.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/eegg-dat-S00289-t1-v1-patch-halo.h5_data_train.H | Scale > t12.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/eegg-dat-S00291-t1-v1-patch-halo.h5_data_train.H | Scale > t13.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/eegg-dat-S00292-t1-v1-patch-halo.h5_data_train.H | Scale > t14.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/eegg-dat-S00293-t1-v1-patch-halo.h5_data_train.H | Scale > t15.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/eegg-dat-S00295-t1-v1-patch-halo.h5_data_train.H | Scale > t16.H
	Histogram dinterv=10 min=0 max=150 < dat_2d/eegg-dat-S00297-t1-v1-patch-halo.h5_data_train.H | Scale > t17.H
	Cat axis=2 t1.H t2.H t3.H t4.H t5.H t6.H t7.H t8.H t9.H | Scale | Graph grid=y min1=0.0 max1=150 min2=0.0 max2=1.0 legend=y curvelabel="S00243:S00233:S00242:S00250:S00239:S00254:S000271:S00275:S00286" | Xtpen &
	Cat axis=2 t10.H t11.H t12.H t13.H t14.H t15.H t16.H t17.H | Scale | Graph grid=y min1=0.0 max1=150 min2=0.0 max2=1.0 legend=y curvelabel="S00287:S00288:S00289:S00291:S00292:S00293:S00295:S00297" | Xtpen &

# Labels
eegg_train1_stats_labels:
	# Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5_labels_train.H | Scale > t1.H
	# Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S00233-t1-v1-patch-halo.h5_labels_train.H | Scale > t2.H
	# Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S00242-t1-v3-patch-halo.h5_labels_train.H | Scale > t3.H
	# Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S00250-t1-v1-patch-halo.h5_labels_train.H | Scale > t4.H
	# Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S00239-t1-v1-patch-halo.h5_labels_train.H | Scale > t5.H
	# Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S00254-t1-v1-patch-halo.h5_labels_train.H | Scale > t6.H
	# Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S000271-t1-v1-patch-halo.h5_labels_train.H | Scale > t7.H
	# Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S00275-t1-v1-patch-halo.h5_labels_train.H | Scale > t8.H
	# Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S00286-t1-v1-patch-halo.h5_labels_train.H | Scale > t9.H
	# Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S00287-t1-v1-patch-halo.h5_labels_train.H | Scale > t10.H
	# Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S00288-t1-v1-patch-halo.h5_labels_train.H | Scale > t11.H
	# Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S00289-t1-v1-patch-halo.h5_labels_train.H | Scale > t12.H
	# Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S00291-t1-v1-patch-halo.h5_labels_train.H | Scale > t13.H
	# Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S00292-t1-v1-patch-halo.h5_labels_train.H | Scale > t14.H
	# Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S00293-t1-v1-patch-halo.h5_labels_train.H | Scale > t15.H
	# Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S00295-t1-v1-patch-halo.h5_labels_train.H | Scale > t16.H
	# Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S00297-t1-v1-patch-halo.h5_labels_train.H | Scale > t17.H
	# Cat axis=2 t1.H t2.H t3.H t4.H t5.H t6.H t7.H t8.H t9.H | Scale | Graph grid=y min1=0.0 max1=200 min2=0.0 max2=1.0 legend=y curvelabel="S00243:S00233:S00242:S00250:S00239:S00254:S000271:S00275:S00286" | Xtpen &
	# Cat axis=2 t10.H t11.H t12.H t13.H t14.H t15.H t16.H t17.H | Scale | Graph grid=y min1=0.0 max1=200 min2=0.0 max2=1.0 legend=y curvelabel="S00287:S00288:S00289:S00291:S00292:S00293:S00295:S00297" | Xtpen &
	Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5_labels_train.H | Scale > t1.H
	Histogram dinterv=10 min=0 max=400 < dat_2d/eegg-dat-S00243-t1-v3-patch-halo_clip.h5_labels_train.H | Scale > t2.H
	Cat axis=2 t1.H t2.H | Scale | Graph grid=y min1=0.0 max1=400 min2=0.0 max2=1.0 legend=y curvelabel="No clip:Clip" | Xtpen &

junkDisp-f%:
	Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00243-t1-v3-patch-halo.h5_labels_train.H > t1.H
	Window3d n3=1 f3=$* < dat_2d/eegg-dat-S00243-t1-v3-patch-halo_clip.h5_labels_train.H > t2.H
	Cat axis=3 t1.H t2.H | Grey color=j newclip=1 grid=y gainpanel=a bclip=0 eclip=400 titles="No clip:Clip" | Xtpen pixmaps=y &
