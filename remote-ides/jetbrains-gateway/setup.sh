#!/bin/bash
# Install the following packages for enabling Jupyter-Lab themes
apt-get install -y sudo
apt-get install -y curl
wget https://download.jetbrains.com/python/pycharm-professional-2021.3.3.tar.gz
tar -zxvf pycharm-professional-2021.3.3.tar.gz

yes "" | ./pycharm-2021.3.3/bin/remote-dev-server.sh  run /workspace/runtime/ \
--ssh-link-host $(/sbin/ip -o -4 addr list eth0 | awk '{print $4}' | cut -d/ -f1) \
--ssh-link-user root --ssh-link-port 2222
echo "ELBO ELBO 💪 Gateway Ready"
