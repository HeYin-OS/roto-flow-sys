import torch, torchvision, sys

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

print("torch =", torch.__version__)
print("torchvision =", torchvision.__version__)
print("python =", sys.version)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    x = torch.rand(1024, 1024, device="cuda")
    y = torch.mm(x, x)
    print("Matmul ok, shape:", y.shape)