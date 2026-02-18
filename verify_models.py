import torch
from models import get_resnet18_plain, get_resnet18_skip

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def verify():
    print("--- Verifying Architectures ---")
    
    plain_model = get_resnet18_plain()
    skip_model = get_resnet18_skip()
    
    x = torch.randn(1, 3, 32, 32)
    
    out_plain = plain_model(x)
    out_skip = skip_model(x)
    
    print(f"Plain Model Output Shape: {out_plain.shape}")
    print(f"Skip Model Output Shape: {out_skip.shape}")
    
    plain_params = count_parameters(plain_model)
    skip_params = count_parameters(skip_model)
    
    print(f"Plain Model Parameters: {plain_params:,}")
    print(f"Skip Model Parameters:  {skip_params:,}")
    
    diff = skip_params - plain_params
    print(f"Difference (Skip - Plain): {diff:,}")
    
    if out_plain.shape == out_skip.shape:
        print("\nSuccess: Both models produce identical output shapes.")
    else:
        print("\nError: Output shapes differ.")

if __name__ == "__main__":
    verify()
