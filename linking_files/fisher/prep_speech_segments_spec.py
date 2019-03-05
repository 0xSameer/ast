# coding: utf-8

import os
import sys
import argparse
import json
import pickle
import re
from tqdm import tqdm
import numpy as np

program_descrp = """
merge kaldi speech segments to map with transcriptions and translations
"""

'''
example:
export SPEECH_FEATS=$PWD/fisher/BNF/SW/dev2/npz/dev2
#/afs/inf.ed.ac.uk/group/project/lowres/work/corpora/fisher_kaldi/fbank
python prep_speech_segments.py -m $SPEECH_FEATS -o $PWD/out/

'''
def map_speech_segments(cat_dict, cat_speech_path, cat_out_path, split=False):
    print("mapping speech data, output in: {0:s}".format(cat_out_path))
    if not os.path.isdir(cat_out_path):
        # creating directory
        print("creating directory")
        os.makedirs(cat_out_path)
    else:
        print("directory exists")

    cnt = 0
    sp_id = ""
    for utt_id in tqdm(cat_dict):
        if sp_id != utt_id.rsplit("-",1)[0]:
            # load new speech file
            sp_id = utt_id.rsplit("-",1)[0]
            sp_data = np.load(os.path.join(cat_speech_path, "{0:s}.np".format(sp_id)))
        # end if

        # check if any data in transcription
        # e.g.: map_dict['fisher_dev']['20051017_220530_275_fsp-B-21']
        # does not have any data in the spanish speech
        if len(cat_dict[utt_id]['es_w']) > 0:
            seg_names = [seg['seg_name'] for seg in cat_dict[utt_id]['seg']]

            utt_data = [sp_data[s] for s in seg_names if s in sp_data]
            missing_files = [s for s in seg_names if s not in sp_data]
            if len(missing_files) > 0:
                print("{0:s} files missing".format(" ".join(missing_files)))
            if len(utt_data) == 0:
                print("{0:s} file missing".format(utt_id))
            else:
                utt_data = np.concatenate(utt_data, axis=0)
                # check if sub-dir needs to be create
                if split:
                    # extract year+month from utt_id
                    # e.g. - 20050927_181232_134_fsp-A-15
                    cat_sub_out_path = os.path.join(cat_out_path,
                                                   utt_id.split('_',1)[0])

                    if not os.path.isdir(cat_sub_out_path):
                        # creating directory
                        print("creating directory")
                        os.makedirs(cat_sub_out_path)
                    np.save(os.path.join(cat_sub_out_path, "{0:s}".format(utt_id)), utt_data)
                else:
                    np.save(os.path.join(cat_out_path, "{0:s}".format(utt_id)), utt_data)

    print("done...")



def main():
    parser = argparse.ArgumentParser(description=program_descrp)
    parser.add_argument('-m','--speech_dir', help='directory containing speech features',
                        required=True)
    parser.add_argument('-o','--out_path', help='output path',
                        required=True)

    parser.add_argument('-s','--spec', help='which sets, split by -; e.g. fisher_dev',
                        required=True)

    args = vars(parser.parse_args())
    speech_dir = args['speech_dir']
    out_path = args['out_path']
    spec = args['spec'].split('-')

    if not os.path.exists(speech_dir):
        print("speech features path given={0:s} does not exist".format(
                                                        speech_dir))
        return 0

    # create output file directory:
    if not os.path.exists(out_path):
        print("{0:s} does not exist. Exiting".format(out_path))
        return 0

    # load map dictionary
    map_dict_path = os.path.join(out_path,'bpe_map.dict')

    if not os.path.exists(map_dict_path):
        print("{0:s} does not exist. Exiting".format(map_dict_path))
        return 0

    print("-"*50)
    print("loading map_dict from={0:s}".format(map_dict_path))
    map_dict = pickle.load(open(map_dict_path, "rb"))
    print("-"*50)

    for cat in spec:
        if not os.path.isdir(os.path.join(speech_dir, cat)):
            print("{0:s} does not exist. Exiting!".format(os.path.join(speech_dir, cat)))
        else:    
            cat_speech_path = os.path.join(speech_dir, cat)
            cat_out_path = os.path.join(out_path, cat)
            split = "train" in cat
            map_speech_segments(map_dict[cat], cat_speech_path, cat_out_path, split)

    print("-"*50)
    print("all done ...")

if __name__ == "__main__":
    main()
