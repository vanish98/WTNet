#!/bin/bash

conda create -n WTNet python=3.8 -y
conda install -n WTNet pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
conda install -n WTNet dglteam::dgl-cuda11.3  -y
conda install -n WTNet pyg::pytorch-scatter=2.0.9
conda install -n WTNet  pyg=2.0.4=*cu* -c pyg
