import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from utils.datasets import TensorDataset, MultiEpochsDataLoader
from utils.tools import AverageMeter, accuracy, get_time, Logger
from utils.transforms import (
    DiffAug,
    mix_aug,
    decode_fn,
    decode_zoom,
    RandomFactorResizedCrop,
)

from utils.load_dataset import load_normalize, load_dataset
from utils.load_model import load_model
from validate.losses_benchmark import compute_loss


def repeat(data, num):
    num_dims = data.dim()
    repeat_dims = [num] + [1] * (num_dims - 1)
    data = data.repeat(repeat_dims)
    return data


def eval_data(
    method_name, 
    criterion_name,
    data_save_dir,
    log_save_dir,
    model_ls,
    tar_model_ls=[None],
    factor=1,
    epochs=1000,
    batch_size=None,
    crop_method="factor",
    mix_type="vanilla",
    dsa_strategy="color_crop_cutout_flip_scale_rotate",
    store_log=True,
    eval_times=3,
    num_val=4,
    zca=False,
    logger_name='evaluation_log',
):
    """Evaluate the distilled dataset"""
    # logger
    if store_log == True:
        logger = Logger(log_save_dir, '{}'.format(logger_name))
        print(log_save_dir, '{}.log'.format(logger_name))
    else:
        logger = print
        

    # distilled dataset
    dataset = torch.load(os.path.join(data_save_dir, "data.pt"))
    dataset, data, target = dataset["dataset"], dataset["data"], dataset[
        "label"]
    logger(
        f"Load condensed data from {data_save_dir}\nData shape: {data.shape}\nLabels shape: {target.shape}"
    )

    # batch size
    if batch_size is None:
        if data.shape[0] > 0 and data.shape[0] <= 10:
            batch_size = 10
        if data.shape[0] > 10 and data.shape[0] <= 500:
            batch_size = 50
        elif data.shape[0] > 500 and data.shape[0] <= 20000:
            batch_size = 100
        elif data.shape[0] > 20000:
            batch_size = 200

    # crop
    # crop = transforms.Compose([])
    # if dataset != "imagenet-1k":
    #     if factor >= 2:
    #         data, target = decode_zoom(data, target, factor)
    #         logger(
    #             f"Dataset is decoded\nData shape: {data.shape}\nLabels shape: {target.shape}"
    #         )
    #     elif factor == 1:
    #         data, target = repeat(data, 4), repeat(target, 4)
    #         logger(
    #             f"Dataset is repeated 4 times\nData shape: {data.shape}\nLabels shape: {target.shape}"
    #         )
    # elif dataset == "imagenet-1k":
    #     if crop_method == "factor":
    #         crop = RandomFactorResizedCrop(factor)
    #     elif crop_method == "random":
    #         crop = transforms.RandomResizedCrop(
    #             size=data.shape[-1],
    #             scale=(1 / factor, 1 / factor),
    #             ratio=(1, 1),
    #         )
    if crop_method == "factor":
        crop = RandomFactorResizedCrop(factor)
    elif crop_method == "random":
        crop = transforms.RandomResizedCrop(
            size=data.shape[-1],
            scale=(1 / factor, 1 / factor),
            ratio=(1, 1),
        )

    # train transforms -- train dataset -- train loader
    # old transforms
    # if dataset == "imagenet-1k":
    #     aug = transforms.Compose([
    #         transforms.RandomResizedCrop(
    #             size=224,
    #             scale=(0.08, 1),
    #             interpolation=InterpolationMode.BILINEAR),
    #         transforms.RandomHorizontalFlip(),
    #     ])
    # else:
    #     aug = transforms.Compose([
    #         transforms.RandomCrop(data.shape[-1], padding=data.shape[-1] // 8),
    #         transforms.RandomHorizontalFlip()
    #     ])
    train_transform = transforms.Compose([
        crop,
        load_normalize(dataset),
        # RandomFactorResizedCrop(factor),
        # aug,
    ])
    train_dataset = TensorDataset(data, target, train_transform)
    train_loader = MultiEpochsDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    # dsa_strategy
    if dsa_strategy is not None:
        augment = DiffAug(strategy=dsa_strategy, batch=False)
    else:
        augment = None


    # teacher models
    for tar_model_name in tar_model_ls:
        # student models
        for model_name in model_ls:
            # test transforms -- test dataset -- test loader
            resize = transforms.Compose([])
            # imagenet and its subsets
            if dataset != "tinyimagenet" and "imagenet" in dataset:
                # for resnet
                resize = transforms.Compose(
                    [transforms.Resize(256),
                     transforms.CenterCrop(224)])
                # for convnet
                if "conv" in model_name:
                    # imagenet 1k
                    if dataset == "imagenet-1k":
                        resize = transforms.Compose(
                            [transforms.Resize(73),
                             transforms.CenterCrop(64)])
                    # subsets
                    else:
                        resize = transforms.Compose([
                            transforms.Resize(146),
                            transforms.CenterCrop(128)
                        ])
            transform_val = transforms.Compose(
                [resize,
                 transforms.ToTensor(),
                 load_normalize(dataset)])
            val_dataset = load_dataset(
                dataset=dataset,
                train=False,
                transform=transform_val,
                shuffle=False
            )
            # zca for eval dataset
            if zca:
                # train transforms -- train dataset
                train_transform = transforms.Compose([
                    crop,
                    transforms.ToTensor(),
                    # load_normalize(dataset),
                    # RandomFactorResizedCrop(factor),
                    # aug,
                ])
                trainset = load_dataset(
                    dataset=dataset,
                    train=True,
                    transform=train_transform,
                )
                # test transforms -- test dataset -- test loader
                transform_val = transforms.Compose([
                    resize,
                    transforms.ToTensor(),
                ])
                val_dataset = load_dataset(
                    dataset=dataset,
                    train=False,
                    transform=transform_val,
                    shuffle=False
                )
                nclass = val_dataset.nclass
                # zca
                import kornia as K
                images = []
                labels = []
                print("Train ZCA")
                for i in range(len(trainset)):
                    im, lab = trainset[i]
                    images.append(im)
                    labels.append(lab)
                images = torch.stack(images, dim=0).to('cuda')
                labels = torch.tensor(labels, dtype=torch.long, device="cpu")
                zca = K.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
                zca.fit(images)
                # zca_images = zca(images).to("cpu")
                # train_dataset = TensorDataset(zca_images, labels)
                images = []
                labels = []
                print("Test ZCA")
                for i in range(len(val_dataset)):
                    im, lab = val_dataset[i]
                    images.append(im)
                    labels.append(lab)
                images = torch.stack(images, dim=0).to('cuda')
                labels = torch.tensor(labels, dtype=torch.long, device="cpu")
                zca_images = zca(images).to("cpu")
                val_dataset = TensorDataset(zca_images, labels)
                val_dataset.nclass = nclass

            val_loader = MultiEpochsDataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                persistent_workers=True,
                num_workers=4,
            )

            best_acc_l = []
            acc_l = []
            for _ in range(eval_times):
                best_acc, acc = train(
                    method_name=method_name,
                    model_name=model_name,
                    dataset=dataset,
                    nclass=val_dataset.nclass,
                    train_loader=train_loader,
                    augment=augment,
                    val_loader=val_loader,
                    criterion_name=criterion_name,
                    tar_model_name=tar_model_name,
                    epochs=epochs,
                    logger=logger,
                    mix_type=mix_type,
                    epoch_print_freq=epochs // num_val,
                )
                best_acc_l.append(best_acc)
                acc_l.append(acc)
            logger(
                f"Evaluate {eval_times} times to train {model_name} on {dataset} with teacher {tar_model_name} => mean, std acc: {np.mean(best_acc_l):.1f} $\pm$ {np.std(best_acc_l):.1f}\n"
            )


def train(
    method_name, 
    model_name,
    dataset,
    nclass,
    train_loader,
    val_loader,
    criterion_name="ce",
    tar_model_name=None,
    augment=None,
    epochs=1000,
    logger=print,
    mix_type="vanilla",
    epoch_print_freq=250,
):
    if method_name == 'datm':
        model = (load_model(
            model_name=model_name,
            dataset=dataset,
            pretrained=False,
            net_norm="instance"
        ).train().to("cuda"))
    else:
        model = (load_model(
            model_name=model_name,
            dataset=dataset,
            pretrained=False,
        ).train().to("cuda"))
        
    optimizer = optim.AdamW(
        model.parameters(),
        0.001,
        weight_decay=1e-4,
    )
    if model_name in ["swin_v2_t", "vit_b_16"]:
        optimizer = optim.AdamW(
            model.parameters(),
            0.0001,
            weight_decay=1e-4,
        )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * epochs // 3, 5 * epochs // 6], gamma=0.2)

    # if dataset == "imagenet-1k":
    #     optimizer = torch.optim.AdamW(
    #         model.parameters(),
    #         lr=0.001,
    #         weight_decay=0.01
    #     )
    #     from torch.optim.lr_scheduler import LambdaLR
    #     import math
    #     scheduler = LambdaLR(optimizer, lambda step: 0.5 * (1. + math.cos(math.pi * step / epochs)) if step <= epochs else 0, last_epoch=-1)

 

    if tar_model_name is not None:
        model_tar = (load_model(
            model_name=tar_model_name,
            dataset=dataset,
            pretrained=True,
        ).eval().to("cuda"))
        for p in model_tar.parameters():
            p.requires_grad = False

    else:
        model_tar = None

    # Load pretrained
    cur_epoch, best_acc1, acc1 = 0, 0, 0

    for epoch in range(cur_epoch + 1, epochs + 1):
        top1, top5, losses = train_epoch(
            train_loader,
            model,
            criterion_name,
            optimizer,
            augment,
            mix_type=mix_type,
            model_tar=model_tar,
            nclass=nclass,
        )

        if epoch % epoch_print_freq == 0:
            logger(
                "{2} (Train) [Epoch {0}/{1}] Top1 {top1.avg:.1f}  Top5 {top5.avg:.1f}  Loss {loss.avg:.3f}"
                .format(epoch,
                        epochs,
                        get_time(),
                        top1=top1,
                        top5=top5,
                        loss=losses))
        if epoch % epoch_print_freq == 0:

            with torch.no_grad():
                acc1, _, _ = validate(
                    epochs,
                    val_loader,
                    model,
                    criterion_name,
                    epoch,
                    logger,
                    model_tar,
                    nclass,
                )

            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1

        scheduler.step()
    logger("{1} (Finish) [Epoch {0}] Best acc {best_acc:.1f}".format(
        epochs, get_time(), best_acc=best_acc1))
    return best_acc1, acc1


def train_epoch(
    train_loader,
    model,
    criterion_name,
    optimizer,
    augment=None,
    mix_type="vanilla",
    model_tar=None,
    nclass=None,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    num_exp = 0

    for i, (input, target) in enumerate(train_loader):
        if train_loader.device == "cpu":
            input = input.cuda()
            target = target.cuda()
            
        ## soft label
        # 
        if len(target.shape) > 1:
            target = target.squeeze(dim=1)

        data_time.update(time.time() - end)
        with torch.no_grad():
            if augment != None:
                input = augment(input)
 
            #input, soft_target = mix_aug(mix_type, input, soft_target)

            if model_tar is not None:
                teacher_target = model_tar(input)
            else:
                if len(target.shape) == 1:
                    teacher_target = F.one_hot(target, num_classes=nclass).float()
                else:
                    teacher_target = target.float()
            
        output = model(input)

        loss = compute_loss(criterion_name, output, teacher_target, target)

        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # time
        batch_time.update(time.time() - end)
        end = time.time()

        num_exp += len(target)

    return top1, top5, losses


def validate(
    epochs,
    val_loader,
    model,
    criterion_name,
    epoch,
    logger=None,
    model_tar=None,
    nclass=None,
):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        if model_tar is not None:
            soft_target = model_tar(input)
        else:
            soft_target = target

        if len(target.shape) == 1:
            soft_target = F.one_hot(target, num_classes=nclass).float()
        else:
            soft_target = target.float()

        loss = compute_loss(criterion_name, output, soft_target, target)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # time
        batch_time.update(time.time() - end)
        end = time.time()

    if logger is not None:
        logger(
            "{2} (Test ) [Epoch {0}/{1}] Top1 {top1.avg:.1f}  Top5 {top5.avg:.1f}  Loss {loss.avg:.3f}"
            .format(epoch,
                    epochs,
                    get_time(),
                    top1=top1,
                    top5=top5,
                    loss=losses))
    return top1.avg, top5.avg, losses.avg