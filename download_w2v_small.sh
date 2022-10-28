#!/bin/bash
if [ ! -f "fairseq_model/wav2vec_small.pt" ]; then
    mkdir -p fairseq_model
    wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt -P fairseq_model/
    wget https://raw.githubusercontent.com/pytorch/fairseq/main/LICENSE -P fairseq_model/
fi

if [ ! -f "pretrained/ckpt_w2vsmall" ]; then
    mkdir -p pretrained
    wget https://zenodo.org/record/6785056/files/ckpt_w2vsmall.tar.gz
    tar -zxvf ckpt_w2vsmall.tar.gz
    mv ckpt_w2vsmall pretrained/
    rm ckpt_w2vsmall.tar.gz
    cp fairseq/LICENSE pretrained/
fi
