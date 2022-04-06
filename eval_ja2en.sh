HOME=/home/tbui
EXPDIR=$PWD

SCRIPTS=${HOME}/mosesdecoder/scripts
DETRUECASER=${SCRIPTS}/recaser/detruecase.perl

src=$1
tgt=$2
GPUS=$3
MODEL_NAME=$4
MODEL=$PWD/models/${MODEL_NAME}/checkpoint_best.pt
DETOK=$SCRIPTS/tokenizer/detokenizer.perl

DATASET=$PWD/data
SUBWORD_DATA=$DATASET/tmp/subword-data
BIN_DATA=$DATASET/tmp/bin-data


########################## Validation dataset #########################################

CUDA_VISIBLE_DEVICES=$GPUS env LC_ALL=en_US.UTF-8 fairseq-interactive $BIN_DATA \
            --input $SUBWORD_DATA/valid.${src} \
            --path $MODEL \
            --beam 5 | tee ${PWD}/results/${MODEL_NAME}/valid_trans_result.${tgt}

grep ^H ${PWD}/results/${MODEL_NAME}/valid_trans_result.${tgt} | cut -f3 > ${PWD}/results/${MODEL_NAME}/valid_trans.${tgt}
python3.6 $EXPDIR/postprocess/subword_decode.py -i ${PWD}/results/${MODEL_NAME}/test_trans.${tgt} -o ${PWD}/results/${MODEL_NAME}/test.${tgt} \
                                                -m $DATASET/tmp/sp.16000.en.model

# detruecase
$DETRUECASER < ${PWD}/results/${MODEL_NAME}/valid_rmvbpe.${tgt} > ${PWD}/results/${MODEL_NAME}/valid_detruecase.${tgt}

# detokenize
env LC_ALL=en_US.UTF-8 perl $DETOK < ${PWD}/results/${MODEL_NAME}/valid_detruecase.${tgt} > ${PWD}/results/${MODEL_NAME}/valid.${tgt}

echo "VALID" >> ${PWD}/results/${MODEL_NAME}/valid_result.txt
env LC_ALL=en_US.UTF-8 perl $PWD/multi-bleu.pl $PWD/data/tmp/normalized/valid.${tgt} < ${PWD}/results/${MODEL_NAME}/valid.${tgt} >> ${PWD}/results/${MODEL_NAME}/valid_result.txt

########################## Test dataset #########################################

CUDA_VISIBLE_DEVICES=$GPUS env LC_ALL=en_US.UTF-8 fairseq-interactive $BIN_DATA \
            --input $SUBWORD_DATA/test.${src} \
            --path $MODEL \
            --beam 5 | tee ${PWD}/results/${MODEL_NAME}/test_trans_result.${tgt}

grep ^H ${PWD}/results/${MODEL_NAME}/test_trans_result.${tgt} | cut -f3 > ${PWD}/results/${MODEL_NAME}/test_trans.${tgt}
python3.6 $EXPDIR/postprocess/subword_decode.py -i ${PWD}/results/${MODEL_NAME}/test_trans.${tgt} -o ${PWD}/results/${MODEL_NAME}/test.${tgt} \
                                                -m $DATASET/tmp/sp.16000.en.model

# detruecase
$DETRUECASER < ${PWD}/results/${MODEL_NAME}/test_rmvbpe.${tgt} > ${PWD}/results/${MODEL_NAME}/test_detruecase.${tgt}

# detokenize
env LC_ALL=en_US.UTF-8 perl $DETOK < ${PWD}/results/${MODEL_NAME}/test_detruecase.${tgt} > ${PWD}/results/${MODEL_NAME}/test.${tgt}

echo "TEST" >> ${PWD}/results/${MODEL_NAME}/test_result.txt
env LC_ALL=en_US.UTF-8 perl $PWD/multi-bleu.pl $PWD/data/tmp/normalized/test.${tgt} < ${PWD}/results/${MODEL_NAME}/test.${tgt} >> ${PWD}/results/${MODEL_NAME}/test_result.txt

