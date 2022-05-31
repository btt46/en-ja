# src=$1
# tgt=$2

HOME=/home/tbui
EXPDIR=$PWD

SCRIPTS=${HOME}/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
TRUECASER_TRAIN=$SCRIPTS/recaser/train-truecaser.perl
TRUECASER=$SCRIPTS/recaser/truecase.perl
BPEROOT=$HOME/subword-nmt/subword_nmt


DATASET=$PWD/data
DATASET_NAME="train valid test"
NORMALIZED_DATA=$DATASET/tmp/normalized
TOKENIZED_DATA=$DATASET/tmp/tok
TRUECASED_DATA=$DATASET/tmp/truecased
SUBWORD_DATA=$DATASET/tmp/subword-data
BIN_DATA=$DATASET/tmp/bin-data

# Making directories
if [ ! -d $DATASET/tmp ]; then
    mkdir -vp $DATASET/tmp
fi

if [ ! -d $NORMALIZED_DATA ]; then
    mkdir -vp $NORMALIZED_DATA
fi

if [ ! -d $TOKENIZED_DATA ]; then
    mkdir -vp $TOKENIZED_DATA
fi

if [ ! -d $TRUECASED_DATA ]; then
    mkdir -vp $TRUECASED_DATA
fi

if [ ! -d $SUBWORD_DATA ]; then
    mkdir -vp $SUBWORD_DATA
fi

if [ ! -d $BIN_DATA ]; then
    mkdir -vp $BIN_DATA
    mkdir -vp $BIN_DATA/train.1
    mkdir -vp $BIN_DATA/train.2
fi

# Normalization
echo "=> Normalizing...."

for set in $DATASET_NAME; do
    python3 ${EXPDIR}/preprocess/normalize.py ${DATASET}/${set}.en \
                                    ${NORMALIZED_DATA}/${set}.en
    cp ${DATASET}/${set}.ja ${NORMALIZED_DATA}/${set}.ja
done


# Tokenization
echo "=> Tokenizing...."
for SET in $DATASET_NAME; do
    env LC_ALL=en_US.UTF-8  $TOKENIZER -l en < ${NORMALIZED_DATA}/${SET}.en > ${TOKENIZED_DATA}/${SET}.en
done

# Truecaser
echo "=>  Truecasing...."
echo "Training for english..."
env LC_ALL=en_US.UTF-8  $TRUECASER_TRAIN --model $DATASET/tmp/truecase-model.en --corpus ${TOKENIZED_DATA}/train.en

for set in $DATASET_NAME; do
    env LC_ALL=en_US.UTF-8 $TRUECASER --model $DATASET/tmp/truecase-model.en < ${TOKENIZED_DATA}/${set}.en > ${TRUECASED_DATA}/${set}.en
done

for set in $DATASET_NAME; do
    mecab -Owakati ${NORMALIZED_DATA}/${set}.ja > ${TRUECASED_DATA}/${set}.ja 
done

# SentencePieceでサブワード化
python3.6 $EXPDIR/preprocess/subword_train.py -i ${TRUECASED_DATA}/train.en -o $DATASET/tmp/sp.16000.en -v 16000
python3.6 $EXPDIR/preprocess/subword_train.py -i ${TRUECASED_DATA}/train.ja -o $DATASET/tmp/sp.16000.ja -v 16000

for lang in en ja; do
    for set in $DATASET_NAME; do
        python3.6 $EXPDIR/preprocess/subword_apply.py -i ${TRUECASED_DATA}/${set}.${lang} -o ${SUBWORD_DATA}/${set}.${lang} -m $DATASET/tmp/sp.16000.${lang}.model
    done
done

cp ${SUBWORD_DATA}/train.en ${SUBWORD_DATA}/train.1.src
cp ${SUBWORD_DATA}/train.ja ${SUBWORD_DATA}/train.1.tgt

cp ${SUBWORD_DATA}/train.en ${SUBWORD_DATA}/train.2.tgt
cp ${SUBWORD_DATA}/train.ja ${SUBWORD_DATA}/train.2.src

cat ${SUBWORD_DATA}/valid.en >> ${SUBWORD_DATA}/valid.src
cat ${SUBWORD_DATA}/valid.ja >> ${SUBWORD_DATA}/valid.src
cat ${SUBWORD_DATA}/valid.ja >> ${SUBWORD_DATA}/valid.tgt
cat ${SUBWORD_DATA}/valid.en >> ${SUBWORD_DATA}/valid.tgt

cat ${SUBWORD_DATA}/test.en >> ${SUBWORD_DATA}/test.src
cat ${SUBWORD_DATA}/test.ja >> ${SUBWORD_DATA}/test.src
cat ${SUBWORD_DATA}/test.ja >> ${SUBWORD_DATA}/test.tgt
cat ${SUBWORD_DATA}/test.en >> ${SUBWORD_DATA}/test.tgt

fairseq-preprocess -s src -t tgt \
			--destdir $BIN_DATA/train.1 \
			--trainpref $SUBWORD_DATA/train.1 \
			--validpref $SUBWORD_DATA/valid \
			--testpref $SUBWORD_DATA/test \
			--workers 32 \
            2>&1 | tee $EXPDIR/logs/preprocess

fairseq-preprocess -s src -t tgt \
			--destdir $BIN_DATA/train.2 \
			--trainpref $SUBWORD_DATA/train.2 \
			--validpref $SUBWORD_DATA/valid \
			--testpref $SUBWORD_DATA/test \
			--workers 32 \
            2>&1 | tee $EXPDIR/logs/preprocess