# import os
# def get_data_dir_name():
# 	list_subdirs = os.listdir(os.getcwd() + '/data/')
# 	print('before cleaned list_subdirs:',  list_subdirs)
# 	list_subdirs = [i for i in list_subdirs if 'lvl' in i]
# 	assert len(list_subdirs) == 1
# 	return list_subdirs[0]

## training params 
classifier_type = 'LEHS'

use_fine_tuned_model = False # set to True if using fined tuned bert model
train_file_name = 'train_reduce' # [train, test_17, test_18]
data_dir = f"data/"
per_gpu_train_batch_size = 10
num_train_epochs = 10

## evaluation params
eval_files = ['test_reduce']
eval_epochs = range(5, num_train_epochs + 1)

use_multi_gpu = False # NOTE: MODIFY pbs file if using multi gpu on hpc
do_visualization = False

train_file = f'{train_file_name}.csv'

do_evaluation = False # deprecated! whether to evaluate models during trianing to see live results
use_multiprocessing = False

bert_model = 'bert-base-chinese'
output_dir = 'outputs/'
report_dir = 'reports/'
cache_dir = 'cache/'
max_steps = -1
max_seq_length = 512
per_gpu_eval_size = 5
learning_rate = 2e-5
random_seed = 42
num_labels = 38

gradient_accumulation_steps = 1
warmup_proportion = 0.1
output_mode = 'classification'
# classification
CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

# file name has been modified
stats = f'use_multi_gpu={use_multi_gpu}, use_fine_tuned_model={use_fine_tuned_model},  total_epochs={num_train_epochs}, train_file={train_file}, max_seq_length={max_seq_length}, per_gpu_train_batch_size={per_gpu_train_batch_size}, per_gpu_eval_batch_size={per_gpu_eval_size}'

#%%
all_label = [] #should be your own label, string list

upper_label = [] #should be your own label, string list

lower_2_upper_label_dict = {} #should be your own label dict, lower_label:upper_label. string

upper_ds_dict = {} #upper label description, upper_label:upper_label_description. string

label_2_ds_dict = {} #lower label description, lower_label:lower_label_description. string
