#!/bin/bash
if [ ! -f "fairseq/wav2vec_small.pt" ]; then
    mkdir -p fairseq
    wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt -P fairseq
    wget https://raw.githubusercontent.com/pytorch/fairseq/main/LICENSE -P fairseq/
fi

if [ ! -f "pretrained/ckpt_w2vsmall" ]; then
    mkdir -p pretrained
    wget https://zenodo.org/record/6785056/files/ckpt_w2vsmall.tar.gz
    tar -zxvf ckpt_w2vsmall.tar.gz
    mv ckpt_w2vsmall pretrained/
    rm ckpt_w2vsmall.tar.gz
    cp fairseq/LICENSE pretrained/
fi
