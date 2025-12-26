import torchvision.models as thmodels
import torch
import torch.nn as nn
from data.utils.models import ConvNet
from data.utils.load_dataset import get_classes

# Ignore warnings caused by the specific Torch version
import warnings

warnings.filterwarnings("ignore")


def load_model(
    model_name="resnet18",
    dataset="cifar10",
    pretrained=True,
    custom_model="",
    classes=[],
    net_norm="batch",
):
    # Converts the dataset to lowercase.
    dataset = dataset.lower()

    def get_model(model_name="resnet18"):
        if "conv" in model_name:
            if dataset in ["cifar10", "cifar100"]:
                size = 32
            elif dataset in ["tinyimagenet", "imagenet-1k"]:
                size = 64
            elif dataset in [
                "imagenet-nette",
                "imagenet-woof",
                "imagenet-100",
                "imagenet-10",
            ]:
                size = 128
            else:
                raise ValueError("Unknown dataset.")

            nclass = len(classes)

            model = ConvNet(
                num_classes=nclass,
                net_norm=net_norm,
                net_act="relu",
                net_pooling="avgpooling",
                net_depth=int(model_name[-1]),
                net_width=128,
                channel=3,
                im_size=(size, size),
            )
        elif model_name == "resnet18_modified":
            model = thmodels.__dict__["resnet18"](pretrained=False)
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            model.maxpool = nn.Identity()
        elif model_name == "resnet101_modified":
            model = thmodels.__dict__["resnet101"](pretrained=False)
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            model.maxpool = nn.Identity()
        else:
            model = thmodels.__dict__[model_name](pretrained=False)

        return model

    def pruning_classifier(model=None, classes=[]):
        try:
            model_named_parameters = [name for name, x in model.named_parameters()]
            for name, x in model.named_parameters():
                if (
                    name == model_named_parameters[-1]
                    or name == model_named_parameters[-2]
                ):
                    x.data = x[classes]
        except:
            print("ERROR in changing the number of classes.")

        return model

    if classes == []:
        classes = get_classes(dataset)

    # "imagenet-100" "imagenet-10" "imagenet-first" "imagenet-nette" "imagenet-woof"
    model = get_model(model_name)
    model = pruning_classifier(model, get_classes(dataset))
    if pretrained:
        if dataset in [
            "imagenet-100",
            "imagenet-10",
            "imagenet-nette",
            "imagenet-woof",
            "tinyimagenet",
            "cifar10",
            "cifar100",
        ]:
            if custom_model != "":
                checkpoint = torch.load(
                    f"./data/pretrain_models/{custom_model}.pth", map_location="cpu"
                )
            else:
                checkpoint = torch.load(
                    f"./data/pretrain_models/{dataset}_{model_name}.pth",
                    map_location="cpu",
                )
                model.load_state_dict(checkpoint["model"])
        elif dataset in ["imagenet-1k"]:
            # Specifically, for reading the pre-trained EfficientNet model, the following modifications are made
            if model_name == "efficientNet-b0":
                from torchvision.models._api import WeightsEnum
                from torch.hub import load_state_dict_from_url

                def get_state_dict(self, *args, **kwargs):
                    kwargs.pop("check_hash")
                    return load_state_dict_from_url(self.url, *args, **kwargs)

                WeightsEnum.get_state_dict = get_state_dict

            if custom_model != "":
                checkpoint = torch.load(
                    f"./data/pretrain_models/{custom_model}.pth", map_location="cpu"
                )
                model.load_state_dict(checkpoint["model"])
            else:
                if model_name in thmodels.__dict__:
                    model = thmodels.__dict__[model_name](pretrained=True)
                else:
                    checkpoint = torch.load(
                        f"./data/pretrain_models/{dataset}_{model_name}.pth",
                        map_location="cpu",
                    )
                    model.load_state_dict(checkpoint["model"])

    model = pruning_classifier(model, classes)
    nclass = len(classes)
    model.nclass = nclass

    return model


def display_children_module(model=None):
    # Iterate through model's children and display layer information
    for idx, (name, module) in enumerate(model.named_children()):
        print(f"Layer {idx}: {name}: {module}")


def get_features(model=None, image=None, hook_indices=[-1]):
    if isinstance(model, ConvNet):
        # Ensure hook_indices are non-negative
        for i in range(len(hook_indices)):
            if hook_indices[i] < 0:
                hook_indices[i] = model.depth + hook_indices[i]

        features, output = model.get_feature(
            image, idx_from=model.depth, idx_to=-1, return_prob=False, return_logit=True
        )
        layer_outputs = []
        for index, feature in enumerate(features):
            if index in hook_indices:
                layer_outputs.append(feature.view(feature.size(0), -1))
        return layer_outputs, output
    else:
        # Ensure hook_indices are non-negative
        for i in range(len(hook_indices)):
            if hook_indices[i] < 0:
                hook_indices[i] = len(list(model.children())) + hook_indices[i]

        # Initialize an empty list to store outputs of the interested layers
        layer_outputs = []
        hook_list = []

        # Define a callback function to capture outputs of the interested layers
        def hook_fn(module, input, output):
            layer_outputs.append(input[0])

        # Register hooks for the interested layers
        for idx, (name, module) in enumerate(model.named_children()):
            if idx in hook_indices:
                hook = module.register_forward_hook(hook=hook_fn)
                hook_list.append(hook)

        # Run the model and obtain outputs of specific layers
        output = model(image)

        for hook in hook_list:
            hook.remove()

        # Return the outputs of specific layers along with the final output
        return layer_outputs, output
