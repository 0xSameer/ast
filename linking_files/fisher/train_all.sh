#!/bin/bash
#
# Copyright 2014  Gaurav Kumar.   Apache 2.0
# Recipe for Fisher/Callhome-Spanish
# Made to integrate KALDI with JOSHUA for end-to-end ASR and SMT

# Modified by Mihaela Stoian


. cmd.sh
. path.sh
# mfccdir=`pwd`/mfcc
mfccdir=fisher_mfcc_13dim
set -e

stage=1

# call the next line with the directory where the Spanish Fisher data is
# (the values below are just an example).  This should contain
# subdirectories named as follows:
# DISC1 DIC2

echo "Train all data"

sfisher_speech=/afs/inf.ed.ac.uk/group/project/lowres/work/corpora/fisher_orig/spanish/LDC2010S01
sfisher_transcripts=/afs/inf.ed.ac.uk/group/project/lowres/work/corpora/fisher_orig/spanish/LDC2010T04

split=local/splits/split_fisher


echo "data prep"

local/fsp_data_prep.sh $sfisher_speech $sfisher_transcripts

utils/fix_data_dir.sh data/local/data/train_all

echo "creating mfccs"

nice steps/make_mfcc.sh --nj 30 --cmd "$train_cmd" data/local/data/train_all exp/make_mfcc/train_all $mfccdir;

utils/fix_data_dir.sh data/local/data/train_all
utils/validate_data_dir.sh data/local/data/train_all

echo "***************** copying data"

cp -r data/local/data/train_all data/train_all

echo "***************** creating splits"
local/create_splits.sh $split


echo "computing cmvn"
# Now compute CMVN stats for the train, dev and test subsets
sets="train dev dev2 test"
for set in $sets
do
	steps/compute_cmvn_stats.sh data/${set} exp/make_mfcc/${set} $mfccdir
done
echo "Done"


