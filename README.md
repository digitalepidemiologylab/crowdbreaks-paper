# Crowdbreaks: Tracking Health Trends using Public Social Media Data and Crowdsourcing
This repository contains all additional code/data/analysis for the paper "Crowdbreaks: Tracking Health Trends using Public Social Media Data and Crowdsourcing".

## Install
```
conda env create -f environment.yml
```

## Download tweets
Generate a set of Twitter API keys and download the tweets using the following command:
```
python download_tweets.py -i ./data/vaccine_sentiment_data.csv -o ./data/tweets.jsonl --consumerkey XXX --consumersecret XXX --accesstoken XXX  --accesssecret XXX
```

## Download vaccine sentiment model
```
wget https://s3.eu-central-1.amazonaws.com/crowdbreaks-dev/binaries/fasttext_v1.ftz -o ./data/fasttext_v1.ftz
```
