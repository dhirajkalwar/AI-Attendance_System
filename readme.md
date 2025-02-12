python -m venv venv
venv\Scripts\activate

pip install -r requirement.txt

wget -O EDSR_x4.pb https://github.com/Saafke/EDSR_Tensorflow/raw/refs/heads/master/models/EDSR_x4.pb

python main.py