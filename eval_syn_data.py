from validate.evaluator_benchmark import eval_data
import os
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Dataset Processing')
parser.add_argument('--method', type=str, default='rded', help='method')
parser.add_argument('--criterion_name', type=str, default='gift', help='criterion_name')
parser.add_argument('--dataset', type=str, default='cifar100', help='dataset')
parser.add_argument('--model', type=str, default='conv3', help='model')
parser.add_argument('--tar_model', type=str, default=None, help='tar_model')
parser.add_argument('--ipc', type=int, default=10, help='ipc')
args = parser.parse_args()


def eval(method, criterion_name, dataset, tar_model, model, ipc):

    if method == 'datm':
        syn_data_save_dir = f'./syn_data/{method}/{dataset}_{tar_model}/ipc{ipc}'
    else:
        syn_data_save_dir = f'./syn_data/{method}/{dataset}_{model}/ipc{ipc}'

    log_save_dir = f'./log/{method}/{dataset}/ipc{ipc}/{tar_model}_{model}/'
    if not os.path.exists(log_save_dir):
        os.makedirs(log_save_dir)

    if dataset == 'imagenet-1k':
        epoch_eval = 300
    else:
        epoch_eval = 1000

    logger_name = f'{criterion_name}_log'
    eval_times = 3


    if method == 'gvbsm':
        eval_data(
            method, 
            criterion_name,
            data_save_dir=syn_data_save_dir,
            log_save_dir=log_save_dir,
            model_ls=[model],
            tar_model_ls=[tar_model], 
            factor=1,
            epochs=epoch_eval,  
            batch_size=None,
            crop_method="factor",
            mix_type="vanilla",
            dsa_strategy="color_crop_cutout_flip_scale_rotate",
            store_log=True,
            eval_times=eval_times,
            num_val=4,
            zca=False,
            logger_name=logger_name,  
        )
    elif method == 'sre2l':
        eval_data(
            method, 
            criterion_name,
            data_save_dir=syn_data_save_dir,
            log_save_dir=log_save_dir,
            model_ls=[model],
            tar_model_ls=[tar_model],  
            factor=1,
            epochs=epoch_eval,  
            batch_size=None,
            crop_method="factor",
            mix_type="vanilla",
            dsa_strategy="color_crop_cutout_flip_scale_rotate",
            store_log=True,
            eval_times=eval_times,
            num_val=4,
            zca=False,
            logger_name=logger_name,  
        )
    elif method == 'rded':
        factor = 1
        if dataset == 'imagenet-1k' and model == 'resnet18':
            factor = 2

        eval_data(
            method, 
            criterion_name,
            data_save_dir=syn_data_save_dir,
            log_save_dir=log_save_dir,
            model_ls=[model],
            tar_model_ls=[tar_model],  
            factor=factor,
            epochs=epoch_eval,  
            batch_size=None,
            crop_method="factor",
            mix_type="vanilla",
            dsa_strategy="color_crop_cutout_flip_scale_rotate",
            store_log=True,
            eval_times=eval_times,
            num_val=4,
            zca=False,
            logger_name=logger_name,  
        )
    elif method == 'datm':
        eval_data(
            method, 
            criterion_name,
            data_save_dir=syn_data_save_dir,
            log_save_dir=log_save_dir,
            model_ls=[model],
            tar_model_ls=[None],
            factor=1,
            epochs=epoch_eval,  
            batch_size=None,
            crop_method="factor",
            mix_type="vanilla",
            dsa_strategy="color_crop_cutout_flip_scale_rotate",
            store_log=True,
            eval_times=eval_times,
            num_val=4,
            zca=True,  
            logger_name=logger_name,  
        )



if __name__ == "__main__":

    if args.tar_model is None:
        tar_model = args.model
    
    print('dd_method_name:', args.method)
    print('criterion_name:', args.criterion_name)
    print("tar_model:", tar_model)
    print('model:', args.model)


    if args.method != 'datm':
        if os.path.exists(
                f"./syn_data/{args.method}/{args.dataset}_{args.model}/ipc{args.ipc}/data.pt"
        ):
            eval(
                method=args.method,
                criterion_name=args.criterion_name,
                dataset=args.dataset,
                tar_model=tar_model,
                model=args.model,
                ipc=args.ipc,
            )
    else:
        if os.path.exists(
                f"./syn_data/{args.method}/{args.dataset}_{tar_model}/ipc{args.ipc}/data.pt"
        ):
            eval(
                method=args.method,
                criterion_name=args.criterion_name,
                dataset=args.dataset,
                tar_model=tar_model,
                model=args.model,
                ipc=args.ipc,
            )