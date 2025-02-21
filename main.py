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

###########################
# 1) From dlg_utils.py
###########################
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

###########################
# 2) LeNet Model (unchanged)
###########################
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

###########################
# 3) DLG Attack (preserving dlg_main.py method)
###########################
class DLGAttack:
    def __init__(self, model: nn.Module, num_iterations=300, lr=0.001):
        self.model = model
        self.num_iterations = num_iterations
        self.lr = lr  # Not used by LBFGS, kept for consistency

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

        # 2) Convert integer labels to one-hot (for CIFAR-10, num_classes=10)
        onehot_labels = label_to_onehot(labels, num_classes=10)

        # 3) Create dummy data and dummy label
        B = labels.size(0)
        dummy_data = torch.randn((B,) + input_shape, device=device, requires_grad=True)
        dummy_label = torch.randn((B, 10), device=device, requires_grad=True)

        # 4) LBFGS optimizer on dummy_data & dummy_label
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
        progress_bar = tqdm(range(self.num_iterations), desc=f"Batch {batch_idx+1} | Optimizing")

        for _ in progress_bar:
            def closure():
                optimizer.zero_grad()
                dummy_pred = self.model(dummy_data)
                # Convert dummy_label to a one-hot distribution via softmax
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                dummy_loss = cross_entropy_for_onehot(dummy_pred, dummy_onehot_label)
                # Compute gradients with respect to model parameters
                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)
                grad_diff = sum((gx - gy).pow(2).sum() for gx, gy in zip(dummy_dy_dx, original_dy_dx))
                grad_diff.backward()
                return grad_diff

            grad_diff_val = optimizer.step(closure)
            progress_bar.set_postfix(loss=f"{grad_diff_val.item():.6f}")

        return dummy_data.detach()

###########################
# 4) Extract Gradients (one-hot based)
###########################
def extract_gradients(model, data, labels, device='cuda'):
    model.zero_grad()
    data = data.to(device)
    labels = labels.to(device)
    onehot_labels = label_to_onehot(labels, num_classes=10)
    outputs = model(data)
    loss = cross_entropy_for_onehot(outputs, onehot_labels)
    gradients = torch.autograd.grad(loss, model.parameters())
    return [g.detach().clone() for g in gradients]

###########################
# 5) Visualization Function
###########################
def visualize_results(original_batch, reconstructed_batch, mse_values, psnr_values, batch_idx, mean, std):
    """
    Generates and saves a single figure showing the ground truth images (top row)
    and the corresponding reconstructed images (bottom row) with MSE and PSNR annotations.
    """
    batch_size = len(original_batch)
    # Clip images to [0,1] for display
    original_batch = torch.clamp(original_batch, 0, 1)
    reconstructed_batch = torch.clamp(reconstructed_batch, 0, 1)

    if batch_size > 1:
        fig, axes = plt.subplots(2, batch_size, figsize=(15, 6))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(15, 6))
        axes = axes.reshape(2, 1)

    for i in range(batch_size):
        # Ground Truth (Top Row)
        axes[0, i].imshow(original_batch[i].permute(1,2,0).cpu().numpy())
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Ground Truth", fontsize=12, fontweight='bold')
        # Reconstructed (Bottom Row)
        axes[1, i].imshow(reconstructed_batch[i].permute(1,2,0).cpu().numpy())
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Reconstructed", fontsize=12, fontweight='bold')
        axes[1, i].set_title(f"MSE: {mse_values[i]:.4f}\nPSNR: {psnr_values[i]:.2f}", fontsize=10)

    fig.suptitle(f"Batch {batch_idx+1}: Ground Truth vs Reconstruction", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = f"reconstruction_batch_{batch_idx+1}.png"
    plt.savefig(save_path)
    print(f"Saved visualization: {save_path}")
    plt.close(fig)

###########################
# 6) Metrics
###########################
def mse_metric(original_data, reconstructed_data):
    return (original_data - reconstructed_data).pow(2).mean().item()

def psnr_metric(original_data, reconstructed_data):
    mse_val = mse_metric(original_data, reconstructed_data)
    if mse_val == 0:
        return 100
    max_pixel = 1.0
    return 20 * log10(max_pixel / (mse_val ** 0.5))

###########################
# 7) Main Attack Execution
###########################
def main():
    parser = argparse.ArgumentParser(description="Data Reconstruction Attack")
    parser.add_argument("--attack", type=str, default="dlg", choices=["dlg"])
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"])
    parser.add_argument("--num_rounds", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu'
    print(f"Using device: {device}")

    model = LeNet()
    model.to(device)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    num_batches = args.num_batches
    num_images_per_batch = args.batch_size
    subset_indices = random.sample(range(len(dataset)), k=num_images_per_batch)
    subset_data = Subset(dataset, subset_indices)
    loader = DataLoader(subset_data, batch_size=num_images_per_batch, shuffle=True)

    batches = []
    for i, (data, labels) in enumerate(loader):
        if i >= num_batches:
            break
        batches.append((data, labels))

    attack = DLGAttack(model, num_iterations=args.num_rounds, lr=args.lr)

    for idx, (data, labels) in enumerate(batches):
        data, labels = data.to(device), labels.to(device)

        # Extract gradients using the one-hot based method
        gradients = extract_gradients(model, data, labels, device=device)

        # Get input shape (B, C, H, W)
        input_shape = data.shape[1:]

        # Run DLG reconstruction
        reconstructed = attack.reconstruct_data(gradients, labels, input_shape, idx, device=device)

        # Compute per-image metrics
        mse_values, psnr_values = [], []
        for i in range(data.size(0)):
            mse_val = mse_metric(data[i], reconstructed[i])
            psnr_val = psnr_metric(data[i], reconstructed[i])
            mse_values.append(mse_val)
            psnr_values.append(psnr_val)

        # Visualize and save ground truth vs. reconstruction in one figure
        visualize_results(data, reconstructed, mse_values, psnr_values, idx, mean=[0,0,0], std=[1,1,1])

if __name__ == "__main__":
    main()
