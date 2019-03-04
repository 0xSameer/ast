#!/bin/bash

echo "MUST BE HAVE ACCESS TO THE MFCC DIR"
# source path.sh should be done first in the Kaldi environment
languages="pl po sw" # no capital letters
sets="train dev test"

for lang in $languages; do
 echo "Start $lang"
 for set in $sets; do
	apply-cmvn --norm-vars=true --utt2spk=ark:gp${lang}/${set}/utt2spk \
		scp:gp${lang}/${set}/cmvn.scp scp:gp${lang}/${set}/feats.scp \
		ark:- | copy-feats ark:- \
		ark,t:gp${lang}/all_${set}_cmvn.ark
 	echo "Done for $lang $set" 
 done
done
