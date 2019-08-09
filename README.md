# MatchPyramid_torch
An pytorch implementation of MatchPyramid "Text Matching as Image Recognition"

## Prepare msrp dataset
download and install dataset from https://www.microsoft.com/en-us/download/details.aspx?id=52398   
mkdir data
mv msr_paraphrase_train.txt ./data
mv msr_paraphrase_test.txt ./data

## Install
mkdir dump  
pip install -r requirements.txt  
wget http://nlp.stanford.edu/data/glove.6B.zip  
unzip glove.6B.zip  
mv glove.6B.300d.txt ./data

## Train and Evaluate
python -m src.main

