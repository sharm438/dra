import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from math import log10
from torch.utils.data import DataLoader
from typing import List, Tuple
from tqdm import tqdm
import random
import torch.nn.functional as F

import torchvision.models as models
import pdb

# ------------------------
# BasicBlock for ResNet-18
# ------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

# ------------------------
# Bottleneck for ResNet-50
# ------------------------
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

# ------------------------
# General ResNet class
# ------------------------
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for st in strides:
            layers.append(block(self.in_planes, planes, st))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Helpers to create specific ResNet variants
def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

# ---------------------------
# Visualization 
# ---------------------------
def visualize_results(original_batch, reconstructed_batch, mse_values, psnr_values, batch_idx, exp_name):
    batch_size = len(original_batch)
    original_batch = torch.clamp(original_batch, 0, 1)
    reconstructed_batch = torch.clamp(reconstructed_batch, 0, 1)
    
    if batch_size > 1:
        fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 3, 6))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(4, 6))
        axes = np.expand_dims(axes, axis=1)  # Make it indexable like a 2D array
    
    for i in range(batch_size):
        axes[0, i].imshow(original_batch[i].permute(1,2,0).cpu().numpy())
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Ground Truth", fontsize=12, fontweight='bold')
        
        axes[1, i].imshow(reconstructed_batch[i].permute(1,2,0).cpu().numpy())
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Reconstructed", fontsize=12, fontweight='bold')
        axes[1, i].set_title(f"MSE: {mse_values[i]:.4f}\nPSNR: {psnr_values[i]:.2f}", fontsize=9)

    plt.tight_layout()
    save_path = f"{exp_name}_batch_{batch_idx+1}.png"
    plt.savefig(save_path)
    print(f"Saved visualization: {save_path}")
    plt.close(fig)

# ---------------------------
# Metrics 
# ---------------------------
def mse_metric(original_data, reconstructed_data):
    return (original_data - reconstructed_data).pow(2).mean().item()

def psnr_metric(original_data, reconstructed_data):
    mse_val = mse_metric(original_data, reconstructed_data)
    if mse_val == 0:
        return 100
    max_pixel = 1.0
    return 20 * log10(max_pixel / (mse_val ** 0.5))

# ---------------------------
# DLG Attack 
# ---------------------------
class DLGAttack:
    """
    DLG uses an L2 gradient difference and LBFGS optimizer.
    """
    def __init__(self, model: nn.Module, num_iterations=300, lr=0.001):
        self.model = model
        self.num_iterations = num_iterations
        self.lr = lr

    def reconstruct_data(
        self,
        original_data,
        gradients: List[torch.Tensor],
        labels: torch.Tensor,
        input_shape: Tuple[int],
        batch_idx: int,
        exp_name = ''
    ):
        device = 'cuda'
        original_dy_dx = [g.detach().clone() for g in gradients]
        onehot_labels = label_to_onehot(labels, num_classes=10)
        B = labels.size(0)
        dummy_data = torch.randn((B,) + input_shape, device=device, requires_grad=True)
        dummy_label = torch.randn((B, 10), device=device, requires_grad=True)
        #optimizer = optim.Adam([dummy_data, dummy_label], lr=self.lr)
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
        progress_bar = tqdm(range(self.num_iterations), desc=f"DLG Batch {batch_idx+1}")

        for _ in progress_bar:
            def closure():
                optimizer.zero_grad()
                dummy_pred = self.model(dummy_data)
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                dummy_loss = cross_entropy_for_onehot(dummy_pred, dummy_onehot_label)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)
                grad_diff = sum((gx - gy).pow(2).sum() for gx, gy in zip(dummy_dy_dx, original_dy_dx))
                grad_diff.backward()
                return grad_diff

            grad_diff_val = optimizer.step(closure)
            progress_bar.set_postfix(loss=f"{grad_diff_val.item():.6f}")

        return dummy_data.detach()

# ---------------------------
# IG Attack 
# ---------------------------
def total_variation(x):
    diff1 = torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    diff2 = torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return diff1 + diff2

class IGAttack:
    """
    Inverting Gradients (IG) Attack:
      min 1 - dot(grad(x), grad(x^*)) / (||grad(x)|| ||grad(x^*)||) + alpha * TV(x)
    """
    def __init__(
        self,
        model: nn.Module,
        max_iterations=24000,
        lr=0.1,
        tv_weight=0,
        device='cuda'
    ):
        self.model = model
        self.model.eval()
        self.max_iterations = max_iterations
        self.lr = lr
        self.tv_weight = tv_weight
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def _idlg_label_guess(self, real_gradients):
        last_grad = real_gradients[-1]
        if len(last_grad.shape) == 2:
            sums = last_grad.sum(dim=1)
        else:
            sums = last_grad
        label_idx = torch.argmin(sums, dim=-1).reshape((1,))
        return label_idx

    def reconstruct_data(self, origianl_data, real_gradients, labels, input_shape, batch_idx=0, exp_name=''):
        if labels is None:
            guessed_label = self._idlg_label_guess(real_gradients).to(self.device)
            print(f"[IG] Recovered label via iDLG: {guessed_label.item()}")
            labels = guessed_label
        else:
            labels = labels.view(-1).to(self.device)

        dummy_data = torch.randn((labels.size(0),) + input_shape, device=self.device, requires_grad=True)
        optimizer = optim.Adam([dummy_data], lr=self.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                int(self.max_iterations // 2.667),
                int(self.max_iterations // 1.6),
                int(self.max_iterations // 1.142),
            ],
            gamma=0.1
        )

        for it in range(self.max_iterations):
            def closure():
                optimizer.zero_grad()
                self.model.zero_grad()
                outputs = self.model(dummy_data)
                loss_ce = self.criterion(outputs, labels)
                grads = torch.autograd.grad(loss_ce, self.model.parameters(), create_graph=True)
                dotprod = 0.0
                norm_x  = 0.0
                norm_r  = 0.0
                for g_cur, g_real in zip(grads, real_gradients):
                    dotprod += (g_cur * g_real).sum()
                    norm_x  += g_cur.pow(2).sum()
                    norm_r  += g_real.pow(2).sum()
                sim_loss = 1.0 - dotprod / (norm_x.sqrt() * norm_r.sqrt() + 1e-9)
                tv_term = self.tv_weight * total_variation(dummy_data)
                total_loss = sim_loss + tv_term
                total_loss.backward()
                return total_loss

            cost_val = optimizer.step(closure)
            scheduler.step()

            if (it+1) % 100 == 0 or it == 0:
                lr_current = scheduler.optimizer.param_groups[0]['lr']
                print(f"[IG] Iter {it+1}/{self.max_iterations}, cost={cost_val.item():.6f}, lr={lr_current}")

        with torch.no_grad():
            dummy_data.clamp_(0, 1)
        return dummy_data.detach()

# ---------------------------
# GradInversion Attack
# ---------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GradInversionAttack:
    """
    Implements GradInversion (GradInv) approach with:
      1) Optional label restoration (Sec.3.2),
      2) Gradient matching (Eqn.3),
      3) Fidelity regularization (Eqn.9) including BN-statistics matching,
      4) Group consistency regularization (Eqn.11) for multiple random seeds (Sec.3.4).
    """

    def __init__(
        self,
        model: nn.Module,
        max_iterations=2000,
        lr=0.1,
        alpha_tv=0,     # weight for total variation
        alpha_l2=0,     # weight for L2-norm regularization
        alpha_bn=0,     # weight for BatchNorm-statistics matching
        alpha_group=0,  # weight for group consistency
        alpha_n = 0,
        group_size=4,      # number of random seeds for multi-path search
        device='cuda'
    ):
        """
        Args:
            model: The neural network (with BatchNorm) to invert from.
            max_iterations: Number of total update steps for reconstructing the batch.
            lr:  Learning rate for the optimizer (e.g., Adam).
            alpha_tv:  Coefficient for total variation penalty.
            alpha_l2:  Coefficient for L2 image penalty.
            alpha_bn:  Coefficient for BatchNorm-statistics matching.
            alpha_group: Coefficient for group consistency penalty.
            group_size:  Number of seeds (random initializations) to optimize jointly.
            device: 'cuda' or 'cpu'.
        """

        self.model = model
        self.max_iterations = max_iterations
        self.lr = lr
        self.alpha_tv = alpha_tv
        self.alpha_l2 = alpha_l2
        self.alpha_bn = alpha_bn
        self.alpha_group = alpha_group
        self.alpha_n = alpha_n
        self.group_size = group_size
        self.device = device

        # We will collect BN statistics (batch mean/var) after each forward pass.
        self._bn_layers = []
        self._batch_means = {}
        self._batch_vars = {}

        # Identify and register forward hooks on each BatchNorm2d module.
        self._register_bn_hooks()

    def _register_bn_hooks(self):
        """
        Register a forward hook on each BatchNorm2d layer so we can capture
        the per-batch mean and variance (for BN-stat matching).
        """
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                self._bn_layers.append(module)
                module.register_forward_hook(self._bn_input_hook)

    def _bn_input_hook(self, module, input, _):
        # 'input' is always a tuple; for BatchNorm2d it's just (x,).
        x = input[0]
        # Mean & var of the BN input
        cur_mean = x.mean(dim=[0, 2, 3])
        cur_var  = x.var(dim=[0, 2, 3], unbiased=False)

        self._batch_means[module] = cur_mean
        self._batch_vars[module]  = cur_var
        # We do NOT return anything here (forward_pre_hook returns None by default).


    def _recover_labels_if_needed(self, real_gradients, num_classes=10):
        """
        If labels is None, attempt to restore the label using the
        'minimum-of-feature-dimension' approach from Sec.3.2.
        For simplicity, we pick a single label (works best for K=1 or
        when each label is distinct).
        """
        # Typically, the second to last param is the Linear weight:
        # shape: [num_classes, feature_dim]
        final_fc_grad = real_gradients[-2]  
        # We'll look across the feature dimension (dim=1) for min values.
        # Then pick the row with largest negativity as the predicted class:
        col_min_vals, _ = torch.min(final_fc_grad, dim=1)  # shape [num_classes]
        # The index with the largest negative magnitude:
        label_index = torch.argmin(col_min_vals).unsqueeze(0)
        print(f"[GradInv] Recovered single label = {label_index.item()}")
        return label_index

    def _compute_grad_match_loss(self, dummy_grad, real_grad):
        """
        L2 distance between dummy_grad and real_grad (summed over all layers).
        """
        loss = 0
        for g1, g2 in zip(dummy_grad, real_grad):
            loss += (g1 - g2).pow(2).sum()
        return loss

    def _compute_bn_stat_loss(self):
        """
        DeepInversion BN-statistics matching:
          sum over BN layers of:
             || batch_mean - running_mean ||^2 + || batch_var - running_var ||^2
        Weighted by alpha_bn.
        """
        bn_loss = 0.0
        for bn_module in self._bn_layers:
            # The module’s official running mean/var
            running_mean = bn_module.running_mean.detach()
            running_var  = bn_module.running_var.detach()

            # Actual batch stats from dummy_data (captured by our hook).
            cur_mean = self._batch_means[bn_module]
            cur_var  = self._batch_vars[bn_module]

            bn_loss += F.mse_loss(cur_mean, running_mean, reduction='sum')
            bn_loss += F.mse_loss(cur_var,  running_var,  reduction='sum')

        return self.alpha_bn * bn_loss

    def _total_variation(self, x):
        """
        Standard total variation over an image batch:
          sum of absolute differences in H and W directions.
        """
        diff1 = torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        diff2 = torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return diff1 + diff2

    def _compute_consensus(self, dummy_data_list):
        """
        Lazy group consistency: pixel-wise averaging of all seeds
        to form a 'consensus' image/batch.
        """
        # dummy_data_list is a Python list of length = group_size,
        # each element shaped [B, C, H, W].
        # Stack them => shape [G, B, C, H, W]
        stacked = torch.stack(dummy_data_list, dim=0)
        # Pixel-wise average along the 'group_size' dimension
        consensus = stacked.mean(dim=0)  # shape [B, C, H, W]
        return consensus

    def reconstruct_data(self, original_data, real_gradients, labels, input_shape, batch_idx=0, exp_name=''):
        # If unknown labels, attempt single-label restoration
        if labels is None:
            guessed_label = self._recover_labels_if_needed(real_gradients)
            labels = guessed_label
        labels = labels.view(-1).to(self.device)
        batch_size = labels.size(0)

        # ---------------------------
        # STEP 1) Initialize multiple seeds
        # ---------------------------
        ### NEW ###
        dummy_data_list = []
        for g in range(self.group_size):
            init = torch.randn(
                (batch_size,) + input_shape,
                device=self.device,
                requires_grad=True
            )
            dummy_data_list.append(init)

        # Convert labels to one-hot
        num_classes = 1000
        onehot = torch.zeros(labels.size(0), num_classes, device=labels.device)
        onehot.scatter_(1, labels.unsqueeze(1), 1.0)
        

        # Single optimizer for all seeds
        optimizer = torch.optim.Adam(dummy_data_list, lr=self.lr)

        print(f"[GradInv] Reconstructing batch {batch_idx+1}, group_size={self.group_size}")

        # Main optimization loop
        for it in range(self.max_iterations):
            optimizer.zero_grad()
            total_loss = 0.0

            # ---------------------------
            # STEP 2) For each seed: compute grad match + fidelity
            # ---------------------------
            for dummy_data in dummy_data_list:
                self.model.train()
                outputs = self.model(dummy_data)
                ce_loss = cross_entropy_for_onehot(outputs, onehot)
                dummy_grad = torch.autograd.grad(ce_loss, self.model.parameters(), create_graph=True)

                grad_loss = self._compute_grad_match_loss(dummy_grad, real_gradients)
                bn_loss   = self._compute_bn_stat_loss()
                tv_loss   = self._total_variation(dummy_data) * self.alpha_tv
                l2_loss   = dummy_data.pow(2).sum() * self.alpha_l2

                total_loss_per_seed = grad_loss + bn_loss + tv_loss + l2_loss
                total_loss = total_loss + total_loss_per_seed

            # ---------------------------
            # STEP 3) Lazy group consistency penalty
            # ---------------------------
            consensus = self._compute_consensus(dummy_data_list)
            group_loss = 0.0
            for dummy_data in dummy_data_list:
                group_loss = group_loss + (dummy_data - consensus).pow(2).sum()
            group_loss = group_loss * self.alpha_group
            total_loss = total_loss + group_loss

            # ---------------------------
            # Backprop + step
            # ---------------------------
            total_loss.backward()
            optimizer.step()

            # with torch.no_grad():
            #     for dummy_data in dummy_data_list:
            #         # Add pixel-wise Gaussian noise scaled by lr * alpha_n
            #         noise = torch.randn_like(dummy_data)
            #         dummy_data.add_(self.lr * self.alpha_n, noise)
            #         dummy_data.clamp_(0, 1)

            if (it+1) % 100 == 0 or it == 0:
                print(f"[GradInv] Iter {it+1}/{self.max_iterations}, total_loss={total_loss.item():.6f}")
                #mse_values = [mse_metric(original_data, dummy_data)]
                #psnr_values = [psnr_metric(original_data, dummy_data)]
                #visualize_results(original_data, dummy_data.cpu().detach().numpy(), mse_values, psnr_values, batch_idx, exp_name)


        # Return the reconstruction from one seed, e.g. seed 0
        return dummy_data_list[0].detach()


# ---------------------------
# Extract gradients (unchanged)
# ---------------------------
def extract_gradients(model, data, labels, num_classes=10, device='cuda'):
    model.zero_grad()
    data = data.to(device)
    labels = labels.to(device)
    onehot_labels = label_to_onehot(labels, num_classes)
    outputs = model(data)
    loss = cross_entropy_for_onehot(outputs, onehot_labels)
    gradients = torch.autograd.grad(loss, model.parameters())
    return [g.detach().clone() for g in gradients]

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Data Reconstruction Attack")
    parser.add_argument("--attack", type=str, required=True, choices=["dlg", "ig", "gi"])
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50"],
                        help="Choose model architecture for the experiment")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet"])
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_rounds", type=int, default=4000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exp", type=str, default="reconstruction", help='output file prefix')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, Attack: {args.attack}, Architecture: {args.arch}")

    # Choose model architecture
    if args.arch == "resnet18":
        model = ResNet18(num_classes=10).to(device)
    else:  # resnet50
        #model = ResNet50(num_classes=10).to(device)
        model = models.__dict__['resnet50']().to(device)
        checkpoint = torch.load('moco_v2_800ep_pretrain.pth.tar', map_location=device)
        from collections import OrderedDict

        # Remove 'module.encoder' prefix if present
        state_dict = checkpoint['state_dict']

        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith("module.encoder_q") and not k.startswith(
                "module.encoder_q.fc"
            ):
                # remove prefix
                state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        
        # Load the adjusted state dictionary into the model
        model.load_state_dict(state_dict, strict=False)
        


    model.eval()

    if args.dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    elif args.dataset == 'imagenet':
    ## load imagenet
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),          # Convert images to tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
        dataset = torchvision.datasets.ImageFolder(root='../imagenet_samples', transform=transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Set LR based on chosen attack (feel free to adjust)
    if args.attack == "dlg":
        lr = 1
        attack = DLGAttack(model, args.num_rounds, lr)
    elif args.attack == "ig":
        lr = 0.1
        attack = IGAttack(model, args.num_rounds, lr, device=device)
    else:  # gi
        lr = 1
        attack = GradInversionAttack(model=model, max_iterations=args.num_rounds, lr=lr, device=device)

    # Run for the specified number of batches
    for idx, (data, labels) in enumerate(loader):
        if idx >= args.num_batches:
            break

        data, labels = data.to(device), labels.to(device)
        
        num_classes = 1000 if args.dataset == 'imagenet' else 10

        # Extract real gradients
        real_gradients = extract_gradients(model, data, labels, num_classes, device=device)
        
        # If using GradInversion, we may optionally do a forward pass on real data
        # to record real BN features. For minimal example:
        if args.attack == "gi":
            with torch.no_grad():
                # Trigger BN forward hooks for real data. This populates real_bn_features
                _ = model(data)

        # Reconstruct
        reconstructed = attack.reconstruct_data(
            data, real_gradients, labels, data.shape[1:], batch_idx=idx, exp_name=args.exp
        )

        # Evaluate MSE/PSNR
        mse_values = [mse_metric(data[i], reconstructed[i]) for i in range(data.size(0))]
        psnr_values = [psnr_metric(data[i], reconstructed[i]) for i in range(data.size(0))]

        # Visualize
        visualize_results(data, reconstructed, mse_values, psnr_values, idx, args.exp)

if __name__ == "__main__":
    main()
