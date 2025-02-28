# =====sre2l (using KL loss)=====
# conv
# cifar100
python eval_syn_data.py --method 'sre2l' --criterion_name 'kl'  --dataset 'cifar100' --model 'conv3' --ipc 10
# tinyimagenet
python eval_syn_data.py --method 'sre2l' --criterion_name 'kl'  --dataset 'tinyimagenet' --model 'conv4' --ipc 10
# imagenet-1k
python eval_syn_data.py --method 'sre2l' --criterion_name 'kl'  --dataset 'imagenet-1k' --model 'conv4' --ipc 10


# resnet
# cifar100
python eval_syn_data.py --method 'sre2l' --criterion_name 'kl'  --dataset 'cifar100' --model 'resnet18_modified' --ipc 10
# tinyimagenet
python eval_syn_data.py --method 'sre2l' --criterion_name 'kl'  --dataset 'tinyimagenet' --model 'resnet18_modified' --ipc 10
# imagenet-1k
python eval_syn_data.py --method 'sre2l' --criterion_name 'kl'  --dataset 'imagenet-1k' --model 'resnet18' --ipc 10


# =====sre2l (using GIFT loss)=====
# conv
# cifar100
python eval_syn_data.py --method 'sre2l' --criterion_name 'gift'  --dataset 'cifar100' --model 'conv3' --ipc 10
# tinyimagenet
python eval_syn_data.py --method 'sre2l' --criterion_name 'gift'  --dataset 'tinyimagenet' --model 'conv4' --ipc 10
# imagenet-1k
python eval_syn_data.py --method 'sre2l' --criterion_name 'gift'  --dataset 'imagenet-1k' --model 'conv4' --ipc 10


# resnet
# cifar100
python eval_syn_data.py --method 'sre2l' --criterion_name 'gift'  --dataset 'cifar100' --model 'resnet18_modified' --ipc 10
# tinyimagenet
python eval_syn_data.py --method 'sre2l' --criterion_name 'gift'  --dataset 'tinyimagenet' --model 'resnet18_modified' --ipc 10
# imagenet-1k
python eval_syn_data.py --method 'sre2l' --criterion_name 'gift'  --dataset 'imagenet-1k' --model 'resnet18' --ipc 10

