#!/usr/bin/env zsh
##
local persistent_init="$(<<"EOF"
###
zmodload zsh/terminfo zsh/system zsh/datetime zsh/mathfunc
###
EOF
)"

print -r -- "$persistent_init" >> ~/.zshenv

eval "$persistent_init"
##
if test -z "$no_conda" && test -n "$gpu_p" ; then
    local conda_opts=()
    if test -n "${conda_usrlocal_p}" ; then
        conda_opts+=( --prefix /usr/local )
    fi

    conda install -y "${conda_opts[@]}" -c rapidsai -c nvidia -c conda-forge \
        python=3.8 rapids=21.12 cudatoolkit=11.0 dask-sql numpy hdbscan xarray \
        || return $?
fi
##
cd ${code_dir} || return $?
git clone https://github.com/Mthrun/FCPS || return $?
export fcps_dir="${PWD}/FCPS"
typeset -p fcps_dir &>> ~/.zshenv || return $?
##
