#!/bin/bash

# source path.sh should be done first in the Kaldi environment
languages="po sw" # no capital letters
sets="train dev test"
cmvn_loc="safe-copy-ast/gp_data_cmvn/"
out_loc="./pretrain_AST_input/"

for lang in $languages; do
 LANG="${lang^^}"        # uppercase lang field
 echo "Start $lang, $LANG"
 mkdir -p ${out_loc}/gp${LANG}
 for set in $sets; do
        #mkdir -p ${out_loc}/gp${LANG}/${set}
	python ./kaldi_io.py \
		${cmvn_loc}/gp${lang}/all_${set}_cmvn.ark \
		${out_loc}/gp${LANG}/${set}

	cp ${cmvn_loc}/gp${lang}/${set}/text ${out_loc}/gp${LANG}/${set}
 	echo "Done for $lang $set" 
 done
done
