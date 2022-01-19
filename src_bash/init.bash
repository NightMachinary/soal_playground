#!/usr/bin/env bash
# @UbuntuOnly
##
sudo apt-get install -y zsh unzip aria2 ncdu htop time
##
wget https://github.com/sharkdp/hyperfine/releases/download/v1.12.0/hyperfine_1.12.0_amd64.deb
sudo dpkg -i hyperfine_1.12.0_amd64.deb
##
