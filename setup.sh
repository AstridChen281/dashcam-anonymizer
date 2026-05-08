echo "127.0.0.1 pypi.ngc.nvidia.com" >> /etc/hosts
pip install ultralytics  --break-system-packages
pip install pybboxes --break-system-packages
# Install opencv-python (code will auto re-encode videos with ffmpeg if OpenCV can't open them)
pip install opencv-python --break-system-packages
# conda install opencv=4.9.0.80
pip uninstall numpy --break-system-packages
pip install numpy==1.26.4  --break-system-packages
pip install natsort --break-system-packages
pip install rich --break-system-packages                
# pip install gdown
# mkdir model
# echo "Downloading the YOLO model..."
# gdown 1uV8IMuGDbmDabdjyeSy4SUKV9OS-ULbe
# mv best.pt model/
echo "Setup complete!"
