# This contains the locations of the tools and data required for running
# the GlobalPhone experiments.

export LC_ALL=C  # For expected sorting and joining behaviour

export KALDI_ROOT="/afs/inf.ed.ac.uk/group/teaching/asr/tools/kaldi"
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh

TEMP="/disk/scratch/mihaela/workspace/kaldi/"
export PATH=$TEMP/../kaldi/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/irstlm/bin/:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1

. $KALDI_ROOT/tools/config/common_path.sh
export LD_LIBRARY_PATH=$KALDI_ROOT/tools/openfst-1.3.4/lib:$KALDI_ROOT/tools/openfst-1.3.4/lib/fst:$KALDI_ROOT/tools/irstlm/lib:$LD_LIBRARY_PATH


KALDISRC=$KALDI_ROOT/src
KALDIBIN=$KALDISRC/bin:$KALDISRC/featbin:$KALDISRC/fgmmbin:$KALDISRC/fstbin
KALDIBIN=$KALDIBIN:$KALDISRC/gmmbin:$KALDISRC/latbin:$KALDISRC/nnetbin
KALDIBIN=$KALDIBIN:$KALDISRC/sgmm2bin:$KALDISRC/lmbin

FSTBIN=$KALDI_ROOT/tools/openfst/bin
LMBIN=$KALDI_ROOT/tools/irstlm/bin

###########
source $KALDI_ROOT/tools/env.sh
##########

[ -d $PWD/local ] || { echo "Error: 'local' subdirectory not found."; }
[ -d $PWD/utils ] || { echo "Error: 'utils' subdirectory not found."; }
[ -d $PWD/steps ] || { echo "Error: 'steps' subdirectory not found."; }

export kaldi_local=$PWD/local
export kaldi_utils=$PWD/utils
export kaldi_steps=$PWD/steps
SCRIPTS=$kaldi_local:$kaldi_utils:$kaldi_steps

export PATH=$PATH:$KALDIBIN:$FSTBIN:$LMBIN:$SCRIPTS

# If the correct version of shorten and sox are not on the path,
# the following will be set by local/gp_check_tools.sh
SHORTEN_BIN=$TEMP/../kaldi/tools/shorten-3.6.1/bin
# e.g. $PWD/tools/shorten-3.6.1/bin
SOX_BIN=$TEMP/../kaldi/tools/sox-14.3.2/bin
# e.g. $PWD/tools/sox-14.3.2/bin
export PATH=$PATH:$SHORTEN_BIN
export PATH=$PATH:$SOX_BIN

SPH2PIPE=$TEMP/../kaldi/tools/tools/sph2pipe_v2.5
export PATH=$PATH:$SPH2PIPE

export PYTHONPATH=$FSTBIN
