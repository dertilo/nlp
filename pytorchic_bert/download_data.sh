
FOLDER=uncased_L-12_H-768_A-12
if ! [ -d $FOLDER ]; then
    wget --trust-server-names https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
    unzip uncased_L-12_H-768_A-12.zip
    rm uncased_L-12_H-768_A-12.zip
fi

GLUE_DIR=glue_data

FOLDER=jiant
if ! [ -d $FOLDER ]; then
    git clone https://github.com/jsalt18-sentence-repl/jiant.git
    python jiant/scripts/download_glue_data.py --data_dir $GLUE_DIR
fi
