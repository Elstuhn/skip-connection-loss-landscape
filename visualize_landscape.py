import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from models import get_resnet56_plain, get_resnet56_skip
import torchvision
import torchvision.transforms as transforms
import os

torch.manual_seed(42)
np.random.seed(42)

def get_params(model):
    return [p.data.clone() for p in model.parameters()]

def set_params(model, params):
    for p, v in zip(model.parameters(), params):
        p.data.copy_(v)

def normalize_direction(direction, weights):
    """
    Filter-wise normalization to ensure directions are invariant to weight scale.
    """
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            d.fill_(0)
        else:
            d.mul_(w.norm() / (d.norm() + 1e-10))

def calculate_loss(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    total = 0
    with torch.no_grad():
        subset_limit = 5 
        for i, (inputs, labels) in enumerate(loader):
            if i >= subset_limit: break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
    return total_loss / total

def generate_landscape_data(model, loader, device, dir_x, dir_y, resolution=40, range_limit=2.5):
    """
    Computes loss values on a grid given fixed directions dir_x and dir_y.
    Wider range to see outside local minima.
    """
    criterion = nn.CrossEntropyLoss()
    weights = get_params(model)
    
    x = np.linspace(-range_limit, range_limit, resolution)
    y = np.linspace(-range_limit, range_limit, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((resolution, resolution))
    
    for i in range(resolution):
        for j in range(resolution):
            perturbed_weights = [
                w + X[i, j] * dx + Y[i, j] * dy
                for w, dx, dy in zip(weights, dir_x, dir_y)
            ]
            set_params(model, perturbed_weights)
            loss_val = calculate_loss(model, loader, device, criterion)
            Z[i, j] = np.log10(loss_val + 1e-6) 
        print(f"Progress: {i+1}/{resolution}")
        
    set_params(model, weights)
    return X, Y, Z

def plot_side_by_side(data_plain, data_skip, save_path):
    fig = plt.figure(figsize=(20, 10))
    
    titles = ["Plain-56 (No Skip Connections)", "Skip-56 (ResNet)"]
    datasets = [data_plain, data_skip]
    
    for i in range(2):
        X, Y, Z = datasets[i]
        
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        
        surf = ax.plot_surface(X, Y, Z, cmap='magma', edgecolor='none', alpha=0.9, antialiased=True)
        
        offset = np.min(Z) - (np.max(Z)-np.min(Z))*0.1
        ax.contourf(X, Y, Z, zdir='z', offset=offset, cmap='magma', alpha=0.3)
        
        ax.set_title(titles[i], fontsize=16, pad=20)
        ax.set_xlabel('Direction X')
        ax.set_ylabel('Direction Y')
        ax.set_zlabel('Log10(Loss)')
        
        ax.view_init(elev=35, azim=45)
        
        fig.colorbar(surf, ax=ax, shrink=0.4, aspect=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    os.makedirs("plots", exist_ok=True)
    num_classes = 100
    resolution = 40
    range_limit = 10.5
    
    plain_model = get_resnet56_plain(num_classes=num_classes).to(device)
    skip_model = get_resnet56_skip(num_classes=num_classes).to(device)

    check_p = "checkpoints/plain56_model_c100.pth"
    check_s = "checkpoints/skip56_model_c100.pth"
    
    if os.path.exists(check_p):
        plain_model.load_state_dict(torch.load(check_p, map_location=device))
    if os.path.exists(check_s):
        skip_model.load_state_dict(torch.load(check_s, map_location=device))
    
    weights_p = get_params(plain_model)
    weights_s = get_params(skip_model)
    
    dir_x_base = [torch.randn_like(p) for p in weights_p]
    dir_y_base = [torch.randn_like(p) for p in weights_p]
    
    dir_x_p = [d.clone() for d in dir_x_base]
    dir_y_p = [d.clone() for d in dir_y_base]
    normalize_direction(dir_x_p, weights_p)
    normalize_direction(dir_y_p, weights_p)
    
    dir_x_s = []
    dir_y_s = []
    for i, p in enumerate(weights_s):
        if i < len(dir_x_base) and p.shape == dir_x_base[i].shape:
            dir_x_s.append(dir_x_base[i].clone())
            dir_y_s.append(dir_y_base[i].clone())
        else:
            dir_x_s.append(torch.randn_like(p))
            dir_y_s.append(torch.randn_like(p))
            
    normalize_direction(dir_x_s, weights_s)
    normalize_direction(dir_y_s, weights_s)
    
    print("\n--- Generating Wide-Range Landscape for Plain-56 ---")
    data_p = generate_landscape_data(plain_model, testloader, device, dir_x_p, dir_y_p, resolution=resolution, range_limit=range_limit)
    
    print("\n--- Generating Wide-Range Landscape for Skip-56 ---")
    data_s = generate_landscape_data(skip_model, testloader, device, dir_x_s, dir_y_s, resolution=resolution, range_limit=range_limit)
    
    print("\n--- Saving Enhanced Comparison Plot ---")
    plot_side_by_side(data_p, data_s, "plots/resnet56_wide_comparison.png")

if __name__ == "__main__":
    main()
