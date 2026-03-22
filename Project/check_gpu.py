import torch
import sys

print("--- System Info ---")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

print("\n--- GPU Info ---")
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version used by Torch: {torch.version.cuda}")
else:
    print("Result: Running on CPU only.")