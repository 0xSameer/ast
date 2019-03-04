#!/bin/bash

# M.C. Stoian; script based on pre-processing text commands providede by S. Bansal
# readme: install scipy in conda env.
languages="PL PO SW"
sets="train dev test"

for lang in $languages; do
 for set in $sets; do
	# remove utterance labels eg. FR34_100 from each line in $set/text 
	# and save the output in fr_$set.fr
	cat gp$lang/$set/text | awk '{$1=""; print $0}' | sed 's/^ *//g' > gp$lang/$set.text

	# keep only utterance labels eg. FR34_100 from each line in $set/text 
	# and save the output in fr_$set.ids
	cat gp$lang/$set/text | cut -d " " -f1 > gp$lang/$set.ids

	# important: fr_train.fr and fr_train.ids must be aligned wrt $set/text

	# post-processing step: remove special characters from fr_train.fr
	sed -e 's/\[[^][]*\]//g' gp$lang/$set.text | sed -e 's/[-_.><=.,!?:~;$@%&]//g' > gp$lang/$set.clean.text
	
	wait;
	echo "Done for $lang $set"
 done
done

