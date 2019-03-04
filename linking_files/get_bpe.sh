#!/bin/bash

# M.C. Stoian; script based on commands provided by S. Bansal and subword-nmt git examples 
# create a examples folder in git, and add all these: instructions2.txt + subword_nmt/instructions.sh

target_loc="../pretrain_AST_input"
languages="PL PO SW"
sets="train dev test"
bpe_limit=1000 
# we apply the learn command below only on TRAIN set
for lang in $languages; do

	current_target_loc=${target_loc}/gp${lang}
	subword-nmt learn-joint-bpe-and-vocab \
		--input ${current_target_loc}/train.clean.text \
		-s ${bpe_limit} \
		-o ${current_target_loc}/codes \
		--write-vocabulary ${current_target_loc}/vocab

	echo "Learned bpe for $lang"

	# then we run apply-bpe on train, dev, test
	for set in $sets; do

		subword-nmt apply-bpe \
			-c ${current_target_loc}/codes \
			--vocabulary ${current_target_loc}/vocab \
			--vocabulary-threshold 1 \
			< ${current_target_loc}/${set}.clean.text \
			> ${current_target_loc}/${set}.BPE_${bpe_limit}

		echo "Applied bpe on $lang $set"
	done
done
# line 88
# check in .fr files no CAPITALS
