HOME=/home/tbui
EXPDIR=$PWD

SCRIPTS=${HOME}/mosesdecoder/scripts
DETRUECASER=${SCRIPTS}/recaser/detruecase.perl

src=en
tgt=ja
GPUS=$1
MODEL_NAME=$2
MODEL=$PWD/models/${MODEL_NAME}/checkpoint_best.pt

DATASET=$PWD/data
BPE_DATA=$DATASET/tmp/bpe-data
BIN_DATA=$DATASET/tmp/bin-data


########################## Validation dataset #########################################

CUDA_VISIBLE_DEVICES=$GPUS env LC_ALL=en_US.UTF-8 fairseq-interactive $BIN_DATA \
            --input $BPE_DATA/valid.${src} \
            --path $MODEL \
            --beam 5 | tee ${PWD}/results/${MODEL_NAME}/valid_trans_result.${tgt}

grep ^H ${PWD}/results/${MODEL_NAME}/valid_trans_result.${tgt} | cut -f3 > ${PWD}/results/${MODEL_NAME}/valid_trans.${tgt}

python3.6 $EXPDIR/postprocess/subword_decode.py -i ${PWD}/results/${MODEL_NAME}/valid_trans.${tgt} -o ${PWD}/results/${MODEL_NAME}/valid.${tgt} \
                                                -m $DATASET/tmp/sp.16000.ja.model

echo "VALID" >> ${PWD}/results/${MODEL_NAME}/valid_result.txt
env LC_ALL=en_US.UTF-8 perl $PWD/multi-bleu.pl $PWD/data/tmp/truecased/valid.${tgt} < ${PWD}/results/${MODEL_NAME}/valid.${tgt} >> ${PWD}/results/${MODEL_NAME}/valid_result.txt

########################## Test dataset #########################################

CUDA_VISIBLE_DEVICES=$GPUS env LC_ALL=en_US.UTF-8 fairseq-interactive $BIN_DATA \
            --input $BPE_DATA/test.${src} \
            --path $MODEL \
            --beam 5 | tee ${PWD}/results/${MODEL_NAME}/test_trans_result.${tgt}

grep ^H ${PWD}/results/${MODEL_NAME}/test_trans_result.${tgt} | cut -f3 > ${PWD}/results/${MODEL_NAME}/test_trans.${tgt}

python3.6 $EXPDIR/postprocess/subword_decode.py -i ${PWD}/results/${MODEL_NAME}/test_trans.${tgt} -o ${PWD}/results/${MODEL_NAME}/test.${tgt} \
                                                -m $DATASET/tmp/sp.16000.ja.model

echo "TEST" >> ${PWD}/results/${MODEL_NAME}/test_result.txt
env LC_ALL=en_US.UTF-8 perl $PWD/multi-bleu.pl $PWD/data/tmp/truecased/test.${tgt} < ${PWD}/results/${MODEL_NAME}/test.${tgt} >> ${PWD}/results/${MODEL_NAME}/test_result.txt

