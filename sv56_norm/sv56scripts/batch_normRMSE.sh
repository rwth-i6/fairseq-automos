#!/bin/sh

# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Xin Wang
# All rights reserved.
# ==============================================================================

### usage:
#   sh batch_normRMSE.sh PATH_TO_DIRECTORY_OF_WAVEFORM
# normalized waveforms will be in the input directory
# 
# note: 
# 1. make sure sox and sv56demo is in your path
# 2. make sure that SCRIPT_DIR points to the directory that contains batch_normRMSE.sh
###

# level of amplitude normalization, 26 by default
LEV=26

# input directory
DATA_DIR=$1

# $2 is root dir
SV56_DIR="$2/sv56_norm"

cd ${DATA_DIR}
for file_name in `ls ./ | grep wav`
do
    bash ${SV56_DIR}/sv56scripts/sub_normRMSE.sh ${file_name} ${SV56_DIR}
done

exit
