################################################################################
################################## Data ########################################
################################################################################
wind2d_c = 80
wind2d_w = 1000

################################## S00243 ######################################
# 1 slice
dat_2d/S00243-dat-t1-v1-64.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 64 --y_patch_size 64 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 30 --skip 0 1 2 4 5 6 7 8 9

dat_2d/S00243-dat-t1-v1-192.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 192 --y_patch_size 192 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 30 --skip 0 1 2 4 5 6 7 8 9

# 1 slice
dat_2d/S00243-dat-t1-v2-64.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 64 --y_patch_size 64 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 30 --skip 0 1 3 4 5 6 7 8 9

dat_2d/S00243-dat-t1-v2-192.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 192 --y_patch_size 192 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 30 --skip 0 1 3 4 5 6 7 8 9

# 1 slice
dat_2d/S00243-dat-t1-v3-64.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 64 --y_patch_size 64 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 30 --skip 0 2 3 4 5 6 7 8 9

dat_2d/S00243-dat-t1-v3-192.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 192 --y_patch_size 192 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 30 --skip 0 2 3 4 5 6 7 8 9

# Full head
dat_2d/S00243-dat-t1-v4-64.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 64 --y_patch_size 64 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 30

dat_2d/S00243-dat-t1-v4-192.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 192 --y_patch_size 192 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 30

# Full head less one slice
dat_2d/S00243-dat-t1-v5-64.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 64 --y_patch_size 64 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 30 --skip 4

dat_2d/S00243-dat-t1-v5-192.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 192 --y_patch_size 192 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 30 --skip 4

# 3 slices
dat_2d/S00243-dat-t1-v6-64.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 64 --y_patch_size 64 --nx_patch 3 --ny_patch 3 --ox_patch 40 --oy_patch 30 --skip 3 4 5 6 7 8 9

dat_2d/S00243-dat-t1-v6-192.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 192 --y_patch_size 192 --nx_patch 1 --ny_patch 1 --ox_patch 40 --oy_patch 30 --skip 3 4 5 6 7 8 9

# Generate datasets
make-dat-t1:
	# make dat_2d/S00243-dat-t1-v1-64.h5 -B
	# make dat_2d/S00243-dat-t1-v1-192.h5 -B
	# make dat_2d/S00243-dat-t1-v2-64.h5 -B
	# make dat_2d/S00243-dat-t1-v2-192.h5 -B
	# make dat_2d/S00243-dat-t1-v3-64.h5 -B
	# make dat_2d/S00243-dat-t1-v3-192.h5 -B
	# make dat_2d/S00243-dat-t1-v4-64.h5 -B
	# make dat_2d/S00243-dat-t1-v4-192.h5 -B
	make dat_2d/S00243-dat-t1-v5-64.h5 -B
	make dat_2d/S00243-dat-t1-v5-192.h5 -B

################################## S00242 ######################################
# Full head
dat_2d/S00242-dat-t1-full-192.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00242/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00242/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 192 --y_patch_size 192 --nx_patch 1 --ny_patch 1 --ox_patch 20 --oy_patch 20

dat_2d/S00242-dat-t1-full-64.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00242/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00242/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 64 --y_patch_size 64 --nx_patch 3 --ny_patch 3 --ox_patch 20 --oy_patch 20

# 1 slice
dat_2d/S00242-dat-t1-v1-64.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00242/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00242/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 64 --y_patch_size 64 --nx_patch 3 --ny_patch 3 --ox_patch 20 --oy_patch 20 --skip 0 2 3 4 5 6 7 8 9

dat_2d/S00242-dat-t1-v1-192.h5:
	./python/CTP_convertDCM_2d.py raw_data/SS00242/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00242/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind2d_c} ${wind2d_w} $@ -v 1 --x_patch_size 192 --y_patch_size 192 --nx_patch 1 --ny_patch 1 --ox_patch 20 --oy_patch 20 --skip 0 2 3 4 5 6 7 8 9
