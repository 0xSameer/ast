"""numpy-kaldi i/o interface
"""
# Author: S. Bansal

import struct
import numpy as np
from tqdm import tqdm
import sys
import os
import pickle

def kalditext2python(textfile, outfolder):
    seg_data = {}
    tmparr = []
    first = True
    arrname = ''
    sys.stderr.flush()
    with open(textfile) as fin:
        with tqdm() as pbar:
            for line in fin:
                splitted = line.strip().split()
                if splitted[-1] == '[':
                    if arrname:
                        seg_data[arrname] = np.array(tmparr).astype(np.float32)
                    if first:
                        arrname = splitted[0]
                        segname = arrname.rsplit("-",2)[0]
                        first = False
                    else:
                        new_arrname = splitted[0]
                        new_segname = new_arrname.rsplit("-",2)[0]

                        if new_arrname != arrname:
                            tmparr = []

                        if new_segname !=  segname:
                            outpath = os.path.join(outfolder, segname+".np")
                            pickle.dump(seg_data, open(outpath, "wb"))
                            pbar.update(1)
                            pbar.set_description("saving: {0:s}".format(outpath))
                            seg_data = {}
                        arrname = new_arrname
                        segname = new_segname
                else:
                    if splitted[-1] == ']':
                        splitted = splitted[:-1]
                    tmparr.append(list(map(float, splitted)))
            seg_data[arrname] = np.array(tmparr).astype(np.float32)
            outpath = os.path.join(outfolder, segname+".np")
            pbar.set_description("saving: {0:s}".format(outpath))
            pickle.dump(seg_data, open(outpath, "wb"))
            pbar.update(1)
    print("done")


def main():
    # my code here
    infile = sys.argv[1]
    outfolder = sys.argv[2]
    print(infile, outfolder)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    if os.path.exists(infile):
        print("haha")
        kalditext2python(infile, outfolder)
        # pickle.dump(data, open(outfile, "wb"))

    else:
        print("file does not exist")

if __name__ == "__main__":
    main()

