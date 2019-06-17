#!/bin/bash
#
# Copyright 2014  Gaurav Kumar.   Apache 2.0
# Recipe for Fisher/Callhome-Spanish
# Made to integrate KALDI with JOSHUA for end-to-end ASR and SMT

# Modified by Mihaela Stoian
# Assumes that train_all.sh has been already executed

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

echo "Compute mfcc np files"

sfisher_speech=/afs/inf.ed.ac.uk/group/project/lowres/work/corpora/fisher_orig/spanish/LDC2010S01
sfisher_transcripts=/afs/inf.ed.ac.uk/group/project/lowres/work/corpora/fisher_orig/spanish/LDC2010T04

echo "apply cmvn; making txt arks; making np files"
sets="train dev dev2 test"
for set in $sets
do
	mkdir ${mfccdir}/txt-arks/
	nice apply-cmvn --norm-vars=true --utt2spk=ark:data/${set}/utt2spk \
		scp:data/${set}/cmvn.scp \
		scp:data/${set}/feats.scp \
		ark:- | copy-feats ark:- \
		ark,t:${mfccdir}/txt-arks/${set}_mfcc.ark


	mkdir -p ${mfccdir}/npz/
	python ../kaldi_io.py \
			${mfccdir}/txt-arks/${set}_mfcc.ark \
			${mfccdir}/npz/${set}

	rm -r ${mfccdir}/txt-arks/
done 
echo "Done"


