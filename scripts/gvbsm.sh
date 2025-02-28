# =====gvbsm (using mse_ce loss)=====
# conv
# cifar100
python eval_syn_data.py --method 'gvbsm' --criterion_name 'mse_ce'  --dataset 'cifar100' --model 'conv3' --ipc 10
# tinyimagenet
python eval_syn_data.py --method 'gvbsm' --criterion_name 'mse_ce'  --dataset 'tinyimagenet' --model 'conv4' --ipc 10
# imagenet-1k
python eval_syn_data.py --method 'gvbsm' --criterion_name 'mse_ce'  --dataset 'imagenet-1k' --model 'conv4' --ipc 10


# resnet
# cifar100
python eval_syn_data.py --method 'gvbsm' --criterion_name 'mse_ce'  --dataset 'cifar100' --model 'resnet18_modified' --ipc 10
# tinyimagenet
python eval_syn_data.py --method 'gvbsm' --criterion_name 'mse_ce'  --dataset 'tinyimagenet' --model 'resnet18_modified' --ipc 10
# imagenet-1k
python eval_syn_data.py --method 'gvbsm' --criterion_name 'mse_ce'  --dataset 'imagenet-1k' --model 'resnet18' --ipc 10


# =====gvbsm (using GIFT loss)=====
# conv
# cifar100
python eval_syn_data.py --method 'gvbsm' --criterion_name 'gift'  --dataset 'cifar100' --model 'conv3' --ipc 10
# tinyimagenet
python eval_syn_data.py --method 'gvbsm' --criterion_name 'gift'  --dataset 'tinyimagenet' --model 'conv4' --ipc 10
# imagenet-1k
python eval_syn_data.py --method 'gvbsm' --criterion_name 'gift'  --dataset 'imagenet-1k' --model 'conv4' --ipc 10


# resnet
# cifar100
python eval_syn_data.py --method 'gvbsm' --criterion_name 'gift'  --dataset 'cifar100' --model 'resnet18_modified' --ipc 10
# tinyimagenet
python eval_syn_data.py --method 'gvbsm' --criterion_name 'gift'  --dataset 'tinyimagenet' --model 'resnet18_modified' --ipc 10
# imagenet-1k
python eval_syn_data.py --method 'gvbsm' --criterion_name 'gift'  --dataset 'imagenet-1k' --model 'resnet18' --ipc 10

