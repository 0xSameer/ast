#!/bin/bash
#
# Copyright 2014  Gaurav Kumar.   Apache 2.0
# Recipe for Fisher/Callhome-Spanish
# Made to integrate KALDI with JOSHUA for end-to-end ASR and SMT

# Modified for BNFs by Mihaela Stoian
# Assumes that the data has been already split and that cmvn stats
# have been computed for each partition (i.e. train, dev, etc.),
# both of which are computed in train_all.sh script

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

echo "Creating BNF archives"

#sfisher_speech=/afs/inf.ed.ac.uk/group/project/lowres/work/corpora/fisher_orig/spanish/LDC2010S01
#sfisher_transcripts=/afs/inf.ed.ac.uk/group/project/lowres/work/corpora/fisher_orig/spanish/LDC2010T04

echo "**************** computing BNFs"
sets="dev train"
dataGP="/disk/scratch/mihaela/workspace/kaldi/data/FR/"
bnf_model="FR"
base_dir="BNF/FR" # assumes contains folder model with "final.raw"
AST_input_dir="../pretrain42_AST_input"
nj=6

source ./path.sh
source activate minf5
for set in $sets
do
	# run forward pass through the BN NN model
	# on the Spanish data set containing the speech;
	# this will not use the Spanish transcripts, just the
	# raw speech; so it is appropriate to use
	# this script for zero-resource settings,
	# where Spanish is our zero-resource language
	mkdir -p ${base_dir}/${set}/archives
	steps/nnet2/dump_bottleneck_features.sh \
		--nj $nj \
		${dataGP}/${set}/ \
		${base_dir}/${set}/bnf_data \
		${base_dir}/model \
		${base_dir}/${set}/archives \
		${base_dir}/${set}/log
	wait	


	echo "Making txt-arks."
	mkdir -p ${base_dir}/${set}/txt_arks/
    	for JOB in $(seq $nj); do
        	copy-feats ark:${base_dir}/${set}/archives/raw_bnfeat_${set}.${JOB}.ark \
	 	   	   ark,t:${base_dir}/${set}/txt_arks/raw_bnfeat_${set}.${JOB}.ark || exit 1;
    	done
	echo "$0: done making txt-arks."
	wait	

	echo "Concatenate txt-arks segments"
	mkdir -p ${base_dir}/${set}/concat-txt-ark
	for JOB in $(seq $nj); do \
		cat ${base_dir}/${set}/txt_arks/raw_bnfeat_${set}.${JOB}.ark;
	done > ${base_dir}/${set}/concat-txt-ark/${set}.ark
	wait
	rm -r ${base_dir}/${set}/txt_arks
	echo "$0: done concatenating txt-arks segments"
	

	echo "Making np files"
	mkdir -p ../io_arks/${bnf_model}/fisher_${set}
	python ../kaldi_io.py \
                ${base_dir}/${set}/concat-txt-ark/${set}.ark \
                ../io_arks/${bnf_model}/${set}

	wait
	rm -r ${base_dir}/${set}/concat-txt-ark
	
	wait
        echo "Done for ${bnf_model} ${set}"
done

echo "All Done"


