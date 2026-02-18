import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models import get_resnet56_plain, get_resnet56_skip
import os

def train_model(model, trainloader, valloader, device, epochs=5, save_path="model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(trainloader):.3f} - Val Acc: {acc:.2f}%')
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
        
        scheduler.step()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    os.makedirs("checkpoints", exist_ok=True)

    num_classes = 100
    epochs = 50

    print(f"\n--- Training ResNet-56 Plain (CIFAR-100, {epochs} epochs) ---")
    plain_model = get_resnet56_plain(num_classes=num_classes).to(device)
    train_model(plain_model, trainloader, testloader, device, epochs=epochs, save_path="checkpoints/plain56_model_c100.pth")

    print(f"\n--- Training ResNet-56 Skip (CIFAR-100, {epochs} epochs) ---")
    skip_model = get_resnet56_skip(num_classes=num_classes).to(device)
    train_model(skip_model, trainloader, testloader, device, epochs=epochs, save_path="checkpoints/skip56_model_c100.pth")

if __name__ == "__main__":
    main()
