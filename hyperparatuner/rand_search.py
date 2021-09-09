import os
from numpy import random
import numpy as np
import shutil
import json
from envs import PROJECT_FOLDER

def remove_all_files(dirpath):
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

def single_task_trial(search_space: dict, rand_seed=42):
    parameter_dict = {}
    for key, value in search_space.items():
        parameter_dict[key] = rand_search_parameter(value)
    parameter_dict['seed'] = rand_seed
    exp_name = 'train.lsg.bs.' + str(parameter_dict['per_gpu_train_batch_size']) + \
               '.lr.' + str(parameter_dict['learning_rate']) + '.sdr.' + str(parameter_dict['sent_drop']) + \
                '.fdr.' + str(parameter_dict['feat_drop']) + '.adr.' + str(parameter_dict['attn_drop']) + \
                '.a.' + str(parameter_dict['alpha']) + '.win.' + str(parameter_dict['window_size']) + \
                '.ln.'+ str(parameter_dict['layers']) + '.hop.' + str(parameter_dict['hop_num']) + '.seed' +str(rand_seed)
    parameter_dict['exp_name'] = exp_name
    return parameter_dict

def rand_search_parameter(space: dict):
    para_type = space['type']
    if para_type == 'fixed':
        return space['value']
    if para_type == 'choice':
        candidates = space['values']
        value = random.choice(candidates, 1).tolist()[0]
        return value
    if para_type == 'range':
        log_scale = space.get('log_scale', False)
        low, high = space['bounds']
        if log_scale:
            value = random.uniform(low=np.log(low), high=np.log(high), size=1)[0]
            value = np.exp(value)
        else:
            value = random.uniform(low=low, high=high, size=1)[0]
        return value
    else:
        raise ValueError('Training batch mode %s not supported' % para_type)

def HypeParameterSpace():
    learning_rate = {'name': 'learning_rate', 'type': 'choice', 'values': [5e-4, 1e-3, 2e-4]} #3e-5, 5e-5, 1e-4, 1.5e-4
    attn_drop_ratio = {'name': 'attn_drop', 'type': 'choice', 'values': [0.3, 0.4, 0.5]}
    feat_drop_ratio = {'name': 'feat_drop', 'type': 'choice', 'values': [0.25, 0.3, 0.4]}
    sent_drop_ratio = {'name': 'sent_drop', 'type': 'choice', 'values': [0.1, 0.15]}
    layers_num = {'name': 'layers', 'type': 'choice', 'values': [3, 6]}
    hop_num = {'name': 'hop_num', 'type': 'choice', 'values': [3, 4, 5]}
    alpha = {'name': 'alpha', 'type': 'choice', 'values': [0.1, 0.15, 0.2]}
    window_size = {'name': 'window_size', 'type': 'choice', 'values': [24]}
    hidden_dim = {'name': 'hidden_dim', 'type': 'choice', 'values': [512]}
    head_num = {'name': 'head_num', 'type': 'choice', 'values': [8]}
    num_train_epochs = {'name': 'num_train_epochs', 'type': 'choice', 'values': [40]}
    per_gpu_train_batch_size = {'name': 'per_gpu_train_batch_size', 'type': 'choice', 'values': [16]}
    optimizer = {'name': 'optimizer', 'type': 'choice', 'values': ['RAdam']}
    lr_scheduler = {'name': 'lr_scheduler', 'type': 'choice', 'values': ['cosine']}
    #++++++++++++++++++++++++++++++++++
    search_space = [learning_rate, per_gpu_train_batch_size, lr_scheduler, optimizer, num_train_epochs,
                    attn_drop_ratio, feat_drop_ratio,
                    sent_drop_ratio, layers_num, hop_num, alpha, window_size, hidden_dim, head_num]
    search_space = dict((x['name'], x) for x in search_space)
    return search_space

def generate_random_search_bash(task_num, seed=42):
    relative_path = PROJECT_FOLDER + '/'
    json_file_path = 'configs/lsgnn/'
    job_path = 'lsgnn_jobs/'
    #================================================
    bash_save_path = relative_path + json_file_path
    os.makedirs(bash_save_path, exist_ok=True)
    jobs_path = relative_path + job_path
    os.makedirs(jobs_path, exist_ok=True)
    # ================================================
    if os.path.exists(jobs_path):
        remove_all_files(jobs_path)
    if os.path.exists(bash_save_path):
        remove_all_files(bash_save_path)
    ##################################################
    search_space = HypeParameterSpace()
    for i in range(task_num):
        parameter_dict = single_task_trial(search_space, seed+i)
        config_json_file_name = 'train.lsg.bs.' + str(parameter_dict['per_gpu_train_batch_size']) + \
               '.lr.' + str(parameter_dict['learning_rate']) + '.sdr.' + str(parameter_dict['sent_drop']) + \
                '.fdr.' + str(parameter_dict['feat_drop']) + '.adr.' + str(parameter_dict['attn_drop']) + \
                '.a.' + str(parameter_dict['alpha']) + '.win.' + str(parameter_dict['window_size']) + \
                '.ln.'+ str(parameter_dict['layers']) + '.hop.' + str(parameter_dict['hop_num']) + '.seed' + \
                str(parameter_dict['seed']) + '.json'
        with open(os.path.join(bash_save_path, config_json_file_name), 'w') as fp:
            json.dump(parameter_dict, fp)
        print('{}\n{}'.format(parameter_dict, config_json_file_name))
        with open(jobs_path + 'lsg_' + config_json_file_name +'.sh', 'w') as rsh_i:
            command_i = "CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 run_experiments.py --config_file " + \
                        json_file_path + config_json_file_name
            rsh_i.write(command_i)
            print('saving jobs at {}'.format(jobs_path + 'lsg_' + config_json_file_name +'.sh'))
    print('{} jobs have been generated'.format(task_num))
