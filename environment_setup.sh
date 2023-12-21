# conda create --name TAO-Amodal python=3.9 -y
# conda activate TAO-Amodal

# CUDA 11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Tao toolkit
pip install git+https://github.com/TAO-Dataset/tao

pip install imageio_ffmpeg

# download detectron2 gcc & g++ ≥ 5.4 are required
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install timm 

# install lvis
pip install lvis


# install these packages to use sort tracker
pip install filterpy==1.4.1
pip install numba
pip install scikit-image==0.14.0
pip install scikit-learn==0.19.1
pip install lap==0.4.0
