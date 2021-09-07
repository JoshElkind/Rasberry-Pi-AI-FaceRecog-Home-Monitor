xhost +SI:localuser:root
export DISPLAY=0:0
#nohup (sudo python3 /home/pi/VSCode/Python/FaceRecogRasPi/Main.py) &
sudo systemctl stop motion
sudo system motion stop
sudo modprobe v4l2loopback exclusive_caps=1 video_nr=2


sudo ffmpeg -i /dev/video0 -f v4l2 -codec:v rawvideo -pix_fmt yuv420p /dev/vide$
sudo python3 /home/pi/VSCode/Python/stream8081_pl.py & >> /s8test.txt 2>&1 
sudo echo "--1--" >> /test_sh.txt
sudo python3 /home/pi/VSCode/Python/FaceRecogRasPi/Main1.py >> /err_stan_file.t$
sudo echo "--3--" >> /test_sh.txt
