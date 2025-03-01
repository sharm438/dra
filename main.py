import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from math import log10
from torch.utils.data import DataLoader, Subset
from typing import List, Tuple
from tqdm import tqdm
import random
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
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

def visualize_results(original_batch, reconstructed_batch, mse_values, psnr_values, batch_idx):
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
    save_path = f"reconstruction_batch_{batch_idx+1}.png"
    plt.savefig(save_path)
    print(f"Saved visualization: {save_path}")
    plt.close(fig)


def mse_metric(original_data, reconstructed_data):
    return (original_data - reconstructed_data).pow(2).mean().item()

def psnr_metric(original_data, reconstructed_data):
    mse_val = mse_metric(original_data, reconstructed_data)
    if mse_val == 0:
        return 100
    max_pixel = 1.0
    return 20 * log10(max_pixel / (mse_val ** 0.5))

# 3) DLG Attack
###########################
class DLGAttack:
    """
    DLG uses an L2 gradient difference and LBFGS optimizer.
    """
    def __init__(self, model: nn.Module, num_iterations=300, lr=0.001):
        self.model = model
        self.num_iterations = num_iterations
        # DLG uses LBFGS with learning rate param
        self.lr = lr

    def reconstruct_data(
        self,
        gradients: List[torch.Tensor],
        labels: torch.Tensor,
        input_shape: Tuple[int],
        batch_idx=0,
        device='cuda'
    ):
        # 1) Save original gradients
        original_dy_dx = [g.detach().clone() for g in gradients]

        # 2) Convert integer labels to one-hot
        onehot_labels = label_to_onehot(labels, num_classes=10)

        # 3) Create dummy data and dummy label
        B = labels.size(0)
        dummy_data = torch.randn((B,) + input_shape, device=device, requires_grad=True)
        dummy_label = torch.randn((B, 10), device=device, requires_grad=True)

        # 4) LBFGS optimizer on dummy_data & dummy_label
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

        progress_bar = tqdm(range(self.num_iterations), desc=f"DLG Batch {batch_idx+1}")

        for _ in progress_bar:
            def closure():
                optimizer.zero_grad()
                dummy_pred = self.model(dummy_data)
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                dummy_loss = cross_entropy_for_onehot(dummy_pred, dummy_onehot_label)
                # L2 difference in gradients:
                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)
                grad_diff = sum((gx - gy).pow(2).sum() for gx, gy in zip(dummy_dy_dx, original_dy_dx))
                grad_diff.backward()
                return grad_diff

            grad_diff_val = optimizer.step(closure)
            progress_bar.set_postfix(loss=f"{grad_diff_val.item():.6f}")

        return dummy_data.detach()

###########################
# 4) IG Attack
###########################
def total_variation(x):
    """
    Isotropic total variation for regularization, summed over the entire batch.
    x shape: (N, C, H, W).
    """
    diff1 = torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    diff2 = torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return diff1 + diff2

class IGAttack:
    """
    Inverting Gradients (IG) Attack:
      min 1 - dot(grad(x), grad(x^*)) / (||grad(x)|| ||grad(x^*)||) + alpha * TV(x)
    Uses Adam + step size decay. Also includes iDLG label reconstruction if label is None.
    """

    def __init__(self,
                 model: nn.Module,
                 max_iterations=24000,
                 lr=0.1,
                 tv_weight=0,
                 device='cuda'):
        self.model = model
        self.model.eval()
        self.max_iterations = max_iterations
        self.lr = lr
        self.tv_weight = tv_weight
        self.device = device

        # We'll define a cross-entropy for known label or iDLG approach if needed
        self.criterion = nn.CrossEntropyLoss()

    def _idlg_label_guess(self, real_gradients):
        """
        iDLG trick for single-image reconstruction:
        label = argmin_j sum( last_weight_gradient[j] )
        If the label is unknown, we pick whichever index is minimal in last layer's gradient sum.
        """
        # The last layer's gradient is real_gradients[-2] or [-1], depending on the network
        # For LeNet, final layer is self.fc3, so it's the last item in real_gradients.
        # Typically you might see a sum across channels. We'll pick argmin along the final layer's bias.
        last_grad = real_gradients[-1]  # shape = (10,) if it's the bias term
        if len(last_grad.shape) == 2:
            # If it's weight, let's check the bias right after. For LeNet, typically the last grad is bias
            # or we can do some custom logic. We'll do a fallback if we see the shape is (10, 84).
            # We'll just pick row sums, then pick argmin
            sums = last_grad.sum(dim=1)
        else:
            # Bias shape typically (10,)
            sums = last_grad
        label_idx = torch.argmin(sums, dim=-1).reshape((1,))
        return label_idx

    def reconstruct_data(self, real_gradients, labels, input_shape):
        """
        Perform the IG attack with Adam & learning rate decay.
        If labels are None, do iDLG label reconstruction.
        """
        # 0) Possibly guess label if not provided
        if labels is None:
            guessed_label = self._idlg_label_guess(real_gradients).to(self.device)
            print(f"[IG] Recovered label via iDLG: {guessed_label.item()}")
            labels = guessed_label
        else:
            labels = labels.view(-1).to(self.device)

        # 1) Initialize dummy data
        dummy_data = torch.randn((labels.size(0),) + input_shape, device=self.device, requires_grad=True)

        # 2) Setup Adam + step size decay
        optimizer = optim.Adam([dummy_data], lr=self.lr)
        # MultiStep decay at ~ 1/3, 1/2, 2/3 of total iterations (like original code)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                int(self.max_iterations // 2.667),
                int(self.max_iterations // 1.6),
                int(self.max_iterations // 1.142),
            ],
            gamma=0.1
        )

        # 3) Iterative update
        for it in range(self.max_iterations):
            def closure():
                optimizer.zero_grad()
                self.model.zero_grad()

                # Forward pass
                outputs = self.model(dummy_data)
                loss_ce = self.criterion(outputs, labels)

                # Compute current gradients w.r.t. model params
                grads = torch.autograd.grad(loss_ce, self.model.parameters(), create_graph=True)

                # --- 1 - dot(...) / (||.|| ||.||) ---
                dotprod = 0.0
                norm_x  = 0.0
                norm_r  = 0.0
                for g_cur, g_real in zip(grads, real_gradients):
                    dotprod += (g_cur * g_real).sum()
                    norm_x  += g_cur.pow(2).sum()
                    norm_r  += g_real.pow(2).sum()
                sim_loss = 1.0 - dotprod / (norm_x.sqrt() * norm_r.sqrt() + 1e-9)

                # Add TV
                tv_term = self.tv_weight * total_variation(dummy_data)

                total_loss = sim_loss + tv_term
                total_loss.backward()
                return total_loss

            cost_val = optimizer.step(closure)
            scheduler.step()

            # Print progress every 50 iters
            if (it+1) % 100 == 0 or it == 0:
                lr_current = scheduler.optimizer.param_groups[0]['lr']
                print(f"[IG] Iter {it+1}/{self.max_iterations}, cost={cost_val.item():.6f}, lr={lr_current}")

        # 4) Clip final solution to [0,1] if you want to enforce x ∈ [0,1]^n
        with torch.no_grad():
            dummy_data.clamp_(0, 1)

        return dummy_data.detach()

def extract_gradients(model, data, labels, device='cuda'):
    """
    Return the ground-truth gradients given a data-label pair.
    Uses cross-entropy with one-hot label for the forward pass
    (consistent with your original DLG approach).
    """
    model.zero_grad()
    data = data.to(device)
    labels = labels.to(device)
    onehot_labels = label_to_onehot(labels, num_classes=10)
    outputs = model(data)
    loss = cross_entropy_for_onehot(outputs, onehot_labels)
    gradients = torch.autograd.grad(loss, model.parameters())
    return [g.detach().clone() for g in gradients]

def cross_entropy_for_onehot(pred, target):
    """
    Computes cross-entropy loss with one-hot target.
    """
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

# -------------- Main Function -------------- #
def main():
    parser = argparse.ArgumentParser(description="Data Reconstruction Attack")
    parser.add_argument("--attack", type=str, required=True, choices=["dlg", "ig"])
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_rounds", type=int, default=4000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)
    model.eval()
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    if args.attack == "dlg":
        lr = 1
    elif args.attack == "ig":
        lr = 0.1
    else:
        raise ValueError("Attack not implemented")
    
    attack = DLGAttack(model, args.num_rounds, lr) if args.attack == "dlg" else IGAttack(model, args.num_rounds, lr, device=device)
    
    for idx, (data, labels) in enumerate(loader):
        if idx >= args.num_batches:
            break
        data, labels = data.to(device), labels.to(device)
        real_gradients = extract_gradients(model, data, labels, device=device)
        reconstructed = attack.reconstruct_data(real_gradients, labels, data.shape[1:])
        
        mse_values = [mse_metric(data[i], reconstructed[i]) for i in range(data.size(0))]
        psnr_values = [psnr_metric(data[i], reconstructed[i]) for i in range(data.size(0))]
        
        visualize_results(data, reconstructed, mse_values, psnr_values, idx)

if __name__ == "__main__":
    main()
