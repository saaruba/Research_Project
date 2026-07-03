#!/bin/bash

set -xe


function add_config_if_not_exist {
    if ! grep -F -q "$1" $HOME/.bashrc; then
        echo "$1" >> $HOME/.bashrc
    fi
}

add_config_if_not_exist "source /opt/ros/humble/setup.bash"
add_config_if_not_exist "source /root/ws_moveit/install/setup.bash"
add_config_if_not_exist "export DISPLAY=:1" 
add_config_if_not_exist "export VNCPORT=5901"


source /opt/ros/humble/setup.bash
source /root/ws_moveit/install/setup.bash

colcon build --symlink-install --continue-on-error || true

LOCAL_SETUP_FILE=`pwd`/install/setup.bash
add_config_if_not_exist "if [ -r $LOCAL_SETUP_FILE ]; then source $LOCAL_SETUP_FILE; fi"

export DISPLAY=:1
export VNCPORT=5901
# 1. Cleanup old locks
sudo rm -f /tmp/.X1-lock /tmp/.X11-unix/X1
sudo rm -rf /tmp/.X11-unix /tmp/.X*-lock /home/lcas/.vnc/*.pid /home/lcas/.vnc/*.log
sudo mkdir -p /tmp/.X11-unix && sudo chmod 1777 /tmp/.X11-unix

# 2. Start system services
sudo service dbus start

# 3. Start VNC server (this listens on 5901)
vncserver :1 -geometry 1280x720 -depth 24 -SecurityTypes None

# 1. Clean up old locks
rm -f /tmp/.X1-lock /tmp/.X11-unix/X1 || true

# 2. Start virtual framebuffer
Xvfb $DISPLAY -screen 0 1920x1080x24 &
sleep 2

# 3. Start XFCE session as lcas user
# We use sudo to ensure it runs in the lcas user context correctly
sudo -u lcas -E startxfce4 &
sleep 3

# 4. Start VNC server (TigervNC)
# Note: we use -localhost no to allow the websockify proxy to connect
sudo -u lcas vncserver $DISPLAY -geometry 1920x1080 -depth 24 -rfbport $VNCPORT -SecurityTypes None &
sleep 2

dbus-launch --exit-with-session startxfce4 &
# 5. Start noVNC proxy
# Maps internal VNC (5901) to the browser-accessible port (5801)
/usr/share/novnc/utils/launch.sh --listen 5801 --vnc localhost:$VNCPORT

sleep 10
DISPLAY=:1 xfconf-query -c xfce4-desktop -p $(xfconf-query -c xfce4-desktop -l | grep "workspace0/last-image") -s /usr/share/backgrounds/xfce/lcas.jpg  || true
