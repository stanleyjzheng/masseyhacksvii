conda create --name masseyhacks python=3.8 anaconda
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
mkdir input
cd input
kaggle datasets download -d cdeotte/jpeg-isic2019-256x256
kaggle datasets download -d cdeotte/jpeg-melanoma-256x256
unzip -q jpeg-isic2019-256x256.zip -d jpeg-isic2019-256x256
unzip -q jpeg-melanoma-256x256.zip -d jpeg-melanoma-256x256
rm *.zip
