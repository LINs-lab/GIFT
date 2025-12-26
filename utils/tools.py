import os
import time
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import save_image

matplotlib.use("Agg")

__all__ = ["Compose", "Lighting", "ColorJitter"]


def dist_l2(data, target):
    dist = (
        (data**2).sum(-1).unsqueeze(1)
        + (target**2).sum(-1).unsqueeze(0)
        - 2 * torch.matmul(data, target.transpose(1, 0))
    )
    return dist


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


class Logger:
    def __init__(self, path, name="log"):
        self.logger = open(os.path.join(path, name + ".txt"), "w")

    def __call__(self, string, end="\n", print_=True):
        if print_:
            print("{}".format(string), end=end)
            if end == "\n":
                self.logger.write("{}\n".format(string))
            else:
                self.logger.write("{} ".format(string))
            self.logger.flush()


class TimeStamp:
    def __init__(self, print_log=True):
        self.prev = time.time()
        self.print_log = print_log
        self.times = {}

    def set(self):
        self.prev = time.time()

    def flush(self):
        if self.print_log:
            print("\n=========Summary=========")
            for key in self.times.keys():
                times = np.array(self.times[key])
                print(
                    f"{key}: {times.sum():.4f}s (avg {times.mean():.4f}s, std {times.std():.4f}, count {len(times)})"
                )
                self.times[key] = []
            self.memory()

            print(f"Peak Memory of GPU: {self.peak_memory_gpu:.4f} GB")
        return self.peak_memory_gpu

    def stamp(self, name=""):
        if self.print_log:
            spent = time.time() - self.prev
            # print(f"{name}: {spent:.4f}s")
            if name in self.times.keys():
                self.times[name].append(spent)
            else:
                self.times[name] = [spent]
            self.set()

    def memory(self):
        memory_allocated_bytes = torch.cuda.max_memory_allocated(device='cuda')
        memory_allocated_gb = memory_allocated_bytes / (1024**3)
        self.peak_memory_gpu = memory_allocated_gb


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if len(target.shape) == 2:
        target = target.argmax(-1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Plotter:
    def __init__(self, path, nepoch, idx=0):
        self.path = path
        self.data = {
            "epoch": [],
            "acc_tr": [],
            "acc_val": [],
            "loss_tr": [],
            "loss_val": [],
        }
        self.nepoch = nepoch
        self.plot_freq = 10
        self.idx = idx

    def update(self, epoch, acc_tr, acc_val, loss_tr, loss_val):
        self.data["epoch"].append(epoch)
        self.data["acc_tr"].append(acc_tr)
        self.data["acc_val"].append(acc_val)
        self.data["loss_tr"].append(loss_tr)
        self.data["loss_val"].append(loss_val)

        if len(self.data["epoch"]) % self.plot_freq == 0:
            self.plot()

    def plot(self, color="black"):
        fig, axes = plt.subplots(1, 4, figsize=(4 * 4, 3))
        fig.tight_layout(h_pad=3, w_pad=3)

        fig.suptitle(f"{self.path}", size=16, y=1.1)

        axes[0].plot(self.data["epoch"], self.data["acc_tr"], color, lw=0.8)
        axes[0].set_xlim([0, self.nepoch])
        axes[0].set_ylim([0, 100])
        axes[0].set_title("acc train")

        axes[1].plot(self.data["epoch"], self.data["acc_val"], color, lw=0.8)
        axes[1].set_xlim([0, self.nepoch])
        axes[1].set_ylim([0, 100])
        axes[1].set_title("acc val")

        axes[2].plot(self.data["epoch"], self.data["loss_tr"], color, lw=0.8)
        axes[2].set_xlim([0, self.nepoch])
        axes[2].set_ylim([0, 3])
        axes[2].set_title("loss train")

        axes[3].plot(self.data["epoch"], self.data["loss_val"], color, lw=0.8)
        axes[3].set_xlim([0, self.nepoch])
        axes[3].set_ylim([0, 3])
        axes[3].set_title("loss val")

        for ax in axes:
            ax.set_xlabel("epochs")

        plt.savefig(f"{self.path}/curve_{self.idx}.png", bbox_inches="tight")
        plt.close()


def save_img(save_dir, img, max_num=100, size=64, nrow=10):
    img = img[:max_num].detach()
    img = torch.clamp(img, min=0.0, max=1.0)

    if img.shape[-1] > size:
        img = F.interpolate(img, size)
    save_image(img.cpu(), save_dir, nrow=nrow)
