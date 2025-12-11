#!/usr/bin/env bash
set -e

echo "AZ-NAS Quick Test Setup"
echo "================================"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python -m venv .venv
else
  echo "Virtual environment already exists"
fi

source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch..."
if ! pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121; then
  echo "CUDA install failed, using default PyTorch"
  pip install torch torchvision
fi

if [ -f "requirements.txt" ]; then
  echo "Installing dependencies..."
  pip install -r requirements.txt
fi

mkdir -p data checkpoints

echo ""
echo "Required: checkpoints/cifar100_teacher.pth"
echo "Optional: checkpoints/cifar10_teacher_ablation.pth"
echo ""

if [ ! -f "checkpoints/cifar100_teacher.pth" ]; then
  echo "ERROR: checkpoints/cifar100_teacher.pth not found"
  exit 1
fi

python - <<EOF
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
EOF

echo ""
echo "Running Quick Test Pipeline"
echo "================================"

echo "[1/3] Loss comparison..."
python 9_aznas_loss_c100.py --quick_test

echo "[2/3] Component ablation..."
python 9_aznas_component_ablation.py --quick_test

echo "[3/3] Overfitting test..."
python 9_aznas_overfitting_test.py --quick_test

echo ""
echo "Complete. Results in checkpoints/*.json"
