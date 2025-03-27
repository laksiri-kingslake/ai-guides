# Speach Recognition with openai/whisper-large-v3-turbo

## NVIDIA Driver Set-up in ubuntu 24.04 (AWS EC2 g4dn)

1. Connect to Your Instance
```bash
ssh -i /path/to/your-key.pem ubuntu@<instance-public-ip>
```

2. Update System Packages
```bash
sudo apt update && sudo apt upgrade -y
```

3. Install Dependencies
```bash
sudo apt install -y build-essential dkms linux-headers-$(uname -r)
```

4. Disable Nouveau Driver
Blacklist Nouveau:
```bash
echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
echo "options nouveau modeset=0" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
```
Update initramfs and reboot:
```bash
sudo update-initramfs -u
sudo reboot
```
5. Reconnect After Reboot
```bash
ssh -i /path/to/your-key.pem ubuntu@<instance-public-ip>
```
6. Add NVIDIA Repository
For Ubuntu 24.04 (Jammy):
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```
Adjust the URL for other Ubuntu versions by replacing ubuntu2204 with your version (e.g., ubuntu2004).

7. Install NVIDIA Driver
Install the latest driver (e.g., 535):
```bash
sudo apt install -y nvidia-driver-535
```

Alternatively, auto-select the recommended driver:
```bash
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
```
8. Reboot the Instance
```bash
sudo reboot
```

9. Verify Installation
After reconnecting, check the GPU status:
```bash
nvidia-smi
```
You should see details of the NVIDIA T4 GPU.

Troubleshooting:
No GPU Detected? Ensure you're using a g4dn.xlarge (or larger) instance type.

Driver Issues? Check logs with dmesg | grep -i nvidia.

Optional: Install CUDA Toolkit
If you need CUDA:
```bash
sudo apt install -y cuda
```

## Development Set-up

1. Set-up python virtual envirnment
```bash
python3 -m venv venv
source ./venv/bin/activate
```

2. requirements.txt
```text
transformers 
gradio
torch
```

3. Install dependencies in requirements
```bash
pip install -r requirements.txt
```

4. Code app.py

5. Run
```bash
python3 app.py
```

## Instructions for Running without GPU

1. Removed torch_dtype=torch.float16 (not needed for CPU)
2. Removed device="cuda:0" (defaults to CPU)
3. Removed return_timestamps (optional, but timestamps add computation)

Important notes:

Install CPU-only PyTorch first (if not already installed):
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

Whisper Large V3 on CPU will be:

Very slow (~10-30x slower than GPU)

Memory intensive (needs ~4GB RAM for short audio)

Better to use smaller model like "openai/whisper-base" for CPU

For better CPU performance, consider adding:

```python
pipe.model.config.forced_decoder_ids = None  # Reduces memory usage
```

## Instructions for openai/widhper-base
Installation First:

```bash
# Install CPU-only requirements

pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install transformers gradio
```
Key Differences from Large V3 Version:

Smaller model (whisper-base vs whisper-large-v3)

No GPU dependencies

Simplified pipeline initialization

Removed optional precision/device parameters

Performance Notes:

Processes ~1 second of audio per second on modern CPUs

Uses ~1GB RAM for short audio (<30s)

For better long-form audio handling, add:

```python
pipe = pipeline(...
    chunk_length_s=30,  # Process in 30s chunks
    stride_length_s=[5, 3]  # Overlap chunks for better continuity
)
```
Optional Optimization:
Add this before launching the app to reduce memory usage:

```python
pipe.model.config.forced_decoder_ids = None
```

## Troubleshooting

1. If getting error: [ValueError: ffmpeg was not found but is required to load audio files from filename]
```bash
sudo apt-get update && sudo apt-get install ffmpeg
```

## Reference
- https://huggingface.co/openai/whisper-large-v3-turbo
