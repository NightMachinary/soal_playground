#!/usr/bin/env bash
# @UbuntuOnly
##
sudo apt-get install -y zsh unzip aria2 ncdu htop time
##
wget https://github.com/sharkdp/hyperfine/releases/download/v1.12.0/hyperfine_1.12.0_amd64.deb
sudo dpkg -i hyperfine_1.12.0_amd64.deb
##
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash ./Miniconda3-latest-*.sh -f -b -p "/usr/local"
#: -f           no error if install prefix already exists

# bash ./Miniconda3-latest-*.sh -b -p "$HOME/miniconda3"
# echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zshenv
##
