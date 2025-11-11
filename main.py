import sys, torch, torchvision
from PySide6.QtWidgets import QApplication

from utils.edge_snapping import EdgeSnappingConfig
from windows.main_window import MainWindow


def print_separator(title):
    print(f"\n{'=' * 50}")
    print(f" {title}")
    print(f"{'=' * 50}")


def test_basic_info():
    print_separator("Basic info")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"torchvision Version: {torchvision.__version__}")
    print(f"CUDA Version on PyTorch Compiling: {torch.version.cuda}")


def test_cuda_availability():
    print_separator("CUDA Support Info")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Availability: {cuda_available}")

    if cuda_available:
        print(f"CUDA Device Number: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name()}")
        print(f"GPU Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("CUDA unavailable，turn to use of CPU")


def main():
    # test_basic_info()
    #
    # test_cuda_availability()

    print_separator("Application Debug Info")

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == '__main__':
    main()
