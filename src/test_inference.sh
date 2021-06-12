pip install -r ../requirements.txt

cd ../input/
kaggle datasets download -d cdeotte/jpeg-melanoma-256x256
unzip -q jpeg-melanoma-256x256.zip -d jpeg-melanoma-256x256

mkdir model
kaggle datasets download -d boliu0/melanoma-winning-models
unzip -q melanoma-winning-models.zip -d model/

rm -r *.zip

cd ../src
python fp16_model.py
python infer.py