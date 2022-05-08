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
SUBWORD_DATA=$DATASET/tmp/subword-data
BIN_DATA=$DATASET/tmp/bin-data


########################## Train dataset #########################################

CUDA_VISIBLE_DEVICES=$GPUS env LC_ALL=en_US.UTF-8 fairseq-interactive $BIN_DATA \
            --input $SUBWORD_DATA/train_10000.${src} \
            --path $MODEL \
            --beam 5 | tee ${PWD}/results/${MODEL_NAME}/train_trans_result.${tgt}

grep ^H ${PWD}/results/${MODEL_NAME}/train_trans_result.${tgt} | cut -f3 > ${PWD}/results/${MODEL_NAME}/train_trans.${tgt}

python3.6 $EXPDIR/postprocess/subword_decode.py -i ${PWD}/results/${MODEL_NAME}/train_trans.${tgt} -o ${PWD}/results/${MODEL_NAME}/train.${tgt} \
                                                -m $DATASET/tmp/sp.16000.ja.model
python3.6 $EXPDIR/preprocess/normalize.py ${PWD}/results/${MODEL_NAME}/train.${tgt} ${PWD}/results/${MODEL_NAME}/train_normalized.${tgt}
echo "train" >> ${PWD}/results/${MODEL_NAME}/train_result.txt
env LC_ALL=en_US.UTF-8 perl $PWD/multi-bleu.pl $PWD/data/tmp/truecased/train_10000.${tgt} < ${PWD}/results/${MODEL_NAME}/train_normalized.${tgt} >> ${PWD}/results/${MODEL_NAME}/train_result.txt
