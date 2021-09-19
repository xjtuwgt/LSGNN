#!/usr/bin/env bash

# DEFINE data related (please make changes according to your configurations)
# DATA ROOT folder where you put data files
DATA_ROOT=./data/

PROCS=${1:-"download"} # define the processes you want to run, e.g. "download,preprocess,train" or "preprocess" only


# Add current pwd to PYTHONPATH
export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 0. Build Database from Wikipedia
download() {
    [[ -d $DATA_ROOT ]] || mkdir -p $DATA_ROOT/models
    wget -P $DATA_ROOT/models/ https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
    wget -P $DATA_ROOT/models/ https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip
    unzip $DATA_ROOT/models/wiki-news-300d-1M.vec.zip -d $DATA_ROOT/models/
    unzip $DATA_ROOT/models/glove.840B.300d.zip -d $DATA_ROOT/models/
}

wiki_download(){
  [[ -d $DATA_ROOT ]] || mkdir -p $DATA_ROOT/knowledge
  wget -P $DATA_ROOT/knowledge/ https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
  tar -xjvf $DATA_ROOT/knowledge/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 -C $DATA_ROOT/knowledge
}

model_download() {
    [[ -d $DATA_ROOT ]] || mkdir -p $DATA_ROOT/models
    wget -P $DATA_ROOT/models/ https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
    wget -P $DATA_ROOT/models/ https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json
}

for proc in "download"
do
    if [[ ${PROCS:-"download"} =~ $proc ]]; then
        $proc
    fi
done
