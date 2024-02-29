wget https://storage.googleapis.com/ads-dataset/subfolder-0.zip -P data
wget https://storage.googleapis.com/ads-dataset/subfolder-1.zip -P data

unzip data/subfolder-0.zip -d data/images/
unzip data/subfolder-1.zip -d data/images/

rm data/subfolder-0.zip
rm data/subfolder-1.zip
