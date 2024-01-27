## Environment
Ubuntu 18.04

python == 3.8

pytorch==1.3.1

torchtext==0.3.1

numpy

tqdm

## Task

introduce a knowledge base into the deep learning model to expand the textual semantics and complete the process of recognizing signs of depression in English-language social media texts, categorizing the texts into three categories: moderate、severe、not depression

## How to run

python main.py --epoch 100 --lr 2e-4 --train_data_path dataset/tagmynews.tsv --txt_embedding_path dataset/glove.6B.300d.txt --cpt_embedding_path dataset/glove.6B.300d.txt  --embedding_dim 300 --train_batch_size 128 --hidden_size 64

## hyperpermeter

utils/config.py

