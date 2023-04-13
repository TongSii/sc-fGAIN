# sc-fGAIN
The imputation method for dropouts in single cell RNA seq, the method is proposed in paper : \\
The files utils.py offers basic function used in both fGAIN_mask.py and fGAIN_mask_Chi.py. \\
fGAIN_mask.py and fGAIN_mask_Chi.py build the main structure of sc-fGAIN's model\\
run_rand_fGAIN_newmask_all.py and run_rand_fGAIN3_newmask_all.py are used to apply model to single cell RNA data sets, with the former one train the whole data and the latter one works on splited data set
