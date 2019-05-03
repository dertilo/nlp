source activate ml

FOLDER=uncased_L-12_H-768_A-12
if ! [ -d $FOLDER ]; then
    wget --trust-server-names https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
    unzip uncased_L-12_H-768_A-12.zip
    rm uncased_L-12_H-768_A-12.zip
fi


export GLUE_DIR=glue_data
export BERT_PRETRAIN=uncased_L-12_H-768_A-12
export SAVE_DIR=saved_models
mkdir $SAVE_DIR


FOLDER=jiant
if ! [ -d $FOLDER ]; then
    git clone https://github.com/jsalt18-sentence-repl/jiant.git
    python jiant/scripts/download_glue_data.py --data_dir $GLUE_DIR
fi

python pytorchic_bert/classify.py \
    --task mrpc \
    --mode train \
    --train_cfg pytorchic_bert/config/train_mrpc.json \
    --model_cfg pytorchic_bert/config/bert_base.json \
    --data_file $GLUE_DIR/MRPC/train.tsv \
    --pretrain_file $BERT_PRETRAIN/bert_model.ckpt \
    --vocab $BERT_PRETRAIN/vocab.txt \
    --save_dir $SAVE_DIR \
    --max_len 128

python pytorchic_bert/classify.py \
    --task mrpc \
    --mode eval \
    --train_cfg pytorchic_bert/config/train_mrpc.json \
    --model_cfg pytorchic_bert/config/bert_base.json \
    --data_file $GLUE_DIR/MRPC/dev.tsv \
    --model_file $SAVE_DIR/model_steps_345.pt \
    --vocab $BERT_PRETRAIN/vocab.txt \
    --max_len 128
