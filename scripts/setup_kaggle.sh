#!/bin/bash
pip install kaggle
mkdir -p ~/.kaggle
# AGENT: Manually place your kaggle.json here first
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d ziya07/depvidmood-facial-expression-video-dataset -p data/raw/ --unzip
echo "DepVidMood ready in data/raw/depvidmood-facial-expression-video-dataset"
