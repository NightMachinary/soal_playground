#!/usr/bin/env zsh

conda install -c rapidsai -c nvidia -c conda-forge \
    rapids=21.12 cudatoolkit=11.5 dask-sql
