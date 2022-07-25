# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

import os
import sys
import argparse
import torch
import torch.nn as nn
import fairseq
from torch.utils.data import DataLoader
from mos_fairseq import MosPredictor, MyDataset
import numpy as np

from i6_utils.helper import bliss_to_tmp_data_dir, text_file_to_tmp_data_dir


def systemID(uttID):
    return uttID.split('-')[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bliss_corpus', type=str, required=False, default=None, help='Path to a bliss corpus')
    parser.add_argument('--text_file', type=str, required=False, default=None, help='Path to a text file containing file names')
    parser.add_argument('--fairseq_base_model', type=str, required=False, default="fairseq/wav2vec_small.pt", help='Path to pretrained fairseq base model.')
    parser.add_argument('--finetuned_checkpoint', type=str, required=False, default="pretrained/ckpt_w2vsmall", help='Path to finetuned MOS prediction checkpoint.')
    parser.add_argument('--outfile', type=str, required=False, default=None, help='Output filename for your answer.txt file')
    args = parser.parse_args()

    if args.bliss_corpus:
        datadir = bliss_to_tmp_data_dir(args.bliss_corpus)
    elif args.text_file:
        datadir = text_file_to_tmp_data_dir(args.text_file)
    else:
        assert False, "Please use either --bliss_corpus or --text_file"
        
    cp_path = args.fairseq_base_model
    my_checkpoint = args.finetuned_checkpoint

    print("Initialize Model", file=sys.stderr)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()

    print('Loading checkpoint', file=sys.stderr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ssl_model_type = cp_path.split('/')[-1]
    if ssl_model_type == 'wav2vec_small.pt':
        SSL_OUT_DIM = 768
    elif ssl_model_type in ['w2v_large_lv_fsh_swbd_cv.pt', 'xlsr_53_56k.pt']:
        SSL_OUT_DIM = 1024
    else:
        print('*** ERROR *** SSL model type ' + ssl_model_type + ' not supported.', file=sys.stderr)
        exit()

    model = MosPredictor(ssl_model, SSL_OUT_DIM).to(device)
    model.eval()

    model.load_state_dict(torch.load(my_checkpoint, map_location=device))

    wavdir = os.path.join(datadir, 'wav')
    validlist = os.path.join(datadir, 'sets/val_mos_list.txt')

    print('Loading data', file=sys.stderr)
    validset = MyDataset(wavdir, validlist)
    validloader = DataLoader(validset, batch_size=1, shuffle=True, num_workers=2, collate_fn=validset.collate_fn)

    total_loss = 0.0
    predictions = { }  # filename : prediction
    criterion = nn.L1Loss()
    print('Starting prediction', file=sys.stderr)

    for i, data in enumerate(validloader, 0):
        inputs, labels, filenames = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        output = outputs.cpu().detach().numpy()[0]
        predictions[filenames[0]] = output  ## batch size = 1

    print(np.mean(predictions.values()))

    ## generate answer.txt for codalab
    if args.outfile:
        ans = open(args.outfile, 'w')
        for k, v in predictions.items():
            outl = k.split('.')[0] + ',' + str(v) + '\n'
            ans.write(outl)
        ans.close()

if __name__ == '__main__':
    main()
