#!/usr/bin/env zsh

conda install -y -c rapidsai -c nvidia -c conda-forge \
    python=3.8 rapids=21.12 cudatoolkit=11.5 dask-sql
