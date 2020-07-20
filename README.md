# Tacotron 2 with IPA and Multispeaker Support

## Applied modifications

I have modified the original tacotron 2 code:
- added multispeaker support
  - in branch `single_speaker_legacy` you find the last working code for the single speaker architecture
- added support to train on THCHS-30 dataset
  - incl. automatically download and convert it to 22,050kHz
- added support to use IPA symbols for training and synthesis
  - incl. automatically convert LJSpeech and THCHS-30 to IPA
  - incl. automatically convert English text into IPA
  - added multiple examples for IPA text
- added support to map weights from a pretrained model
  - e.g. train on LJSpeech and then map weights to new model which trains on THCHS-30 to simulate chinese English accent
  - incl. map for initializing chinese only IPA symbols with English only IPA symbols
- added downloader for LJSpeech dataset
- added plotting of symbolspace in 2D and 3D and in a table
- adjusted paths

# Setup

## Locally with remote Server

Serveraddress for example `joedoe@example.com`.

### SSH login

Execute locally:
```bash
# generate ssh key
ssh-keygen -f ~/.ssh/abc-key-ecdsa -t ecdsa -b 521
# copy the public key to the server
ssh-copy-id -i ~/.ssh/abc-key-ecdsa joedoe@example.com
# connect
ssh -i ~/.ssh/abc-key-ecdsa joedoe@example.com
```

### samba access to get synthesized files

```bash
sudo apt-get update
sudo apt-get install samba
sudo smbpasswd -a user # example set pwd to 123456
sudo nano /etc/samba/smb.conf
```
now add on end:
```txt
[joedoe]
path = /home/joedoe
valid users = joedoe
read only = no
```
and then:
```bash
sudo service smbd restart
```
and then you can mount that drive with:
```bash
mkdir -p joedoe_home
sudo mount -t cifs -o user=joedoe,password=123456,uid=$(id -u),gid=$(id -g) //example.com/joedoe joedoe_home
```

## Create Google Cloud Platform VM (optional)

### Prerequisites

You need to upgrade your free account to a paid account (you retain your free money).
For this training you approximately need 100$.
You also need to inclease your GPU quota at least for the T4 GPU to one.
You can also use an other GPU if you want, the code is optimized for a GPU with 16GB ram.

### Create VM

Create VM with pytorch 1.4 and Cuda 10.1 already installed and configured using *Google Cloud Shell*:

```
export IMAGE_FAMILY="pytorch-1-4-cu101"
export ZONE="europe-west4-b"
export INSTANCE_NAME="tacotron2-instance"
export INSTANCE_TYPE="n1-standard-4"
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --image-family=$IMAGE_FAMILY \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --machine-type=$INSTANCE_TYPE \
    --boot-disk-size=120GB \
    --metadata="install-nvidia-driver=True"
```

- [More information on templates](https://cloud.google.com/ai-platform/deep-learning-vm/docs/quickstart-cli)
- [More information on the parameters](https://cloud.google.com/sdk/gcloud/reference/compute/instances/create)

Install filezilla on your machine to access files:
```bash
sudo apt-get install filezilla -y
# i don't tested if the '-C joedoe' is necessary and if i can use ecdsa
ssh-keygen -f ~/.ssh/gloud-rsa -t rsa -b 4096 -C joedoe
```
and then copy the content of the file `~/.ssh/gloud-rsa.pub` to properties -> SSH of your VM

## Checkout repo

```bash
# get link from https://www.anaconda.com/products/individual and then 'Linux Python 3.7 64-Bit (x86) Installer' copy link
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
# reopen shell
rm Anaconda3-2020.02-Linux-x86_64.sh

git clone https://github.com/stefantaubert/tacotron2
cd tacotron2
conda create -n taco2pytorch python=3.6 -y
conda activate taco2pytorch
pip install -r requirements.txt
```

to be able to run training without being connected with ssh:
```bash
sudo apt install screen
# usage: screen
```

check cuda is installed:
```bash
nvcc --version
```

## IPA synthesis using LJSpeech-1.1 dataset

### Install flite

If you want to train on IPA-Symbols you need to install [flite](https://github.com/festvox/flite) for G2P conversion of English text:

```bash
cd
git clone https://github.com/festvox/flite.git
cd flite
./configure && make
sudo make install
cd testsuite
make lex_lookup
sudo cp lex_lookup /usr/local/bin
```

### Automatically download and prepare dataset

duration: about 1.5h

```bash
```

### Start training

duration: about 5 days on t4

```bash
```

### Continue training

```bash
```

### Synthesize example

```bash
```

## Filestructure

```
$base_dir
├── pretrained
│  ├── waveglow_256channels_universal_v5.pt
├── datasets
│  ├── LJSpeech-1.1
│  ├── THCHS-30
│  ├── THCHS-30-22050kHz
│  ├── ...
├── ds
│  ├── ljs_ipa
│  │  ├── 1
│  │  │  ├── symbols.json
│  │  │  └── filelist.csv
│  ├── thchs_ipa
│  │  ├── A11
│  │  │  ├── symbols.json
│  │  │  └── filelist.csv
│  │  ├── ...
│  ├── ...
├── training_2020-06-17_11-11-03
│  ├── logs
│  │  └── log.txt
│  ├── inference
│  │  ├── 2020-06-18_11-11-03_democritus_A11_500
│  │  │  ├── config.log
│  │  │  ├── input.txt
│  │  │  ├── input_sentences.txt
│  │  │  ├── input_sentences_mapped.txt
│  │  │  ├── input_symbols.txt
│  │  │  ├── input_map.json
│  │  │  ├── 2020-06-17_18-11-03_democritus_A11_500.png
│  │  │  └── 2020-06-17_18-11-03_democritus_A11_500.wav
│  │  ├── ...
│  │  ├── validation_2020-07-14_11-43-47_D11_906_50_9
│  │  │  ├── 2020-07-14_11-43-47_D11_906_50_9_orig.wav
│  │  │  ├── 2020-07-14_11-43-47_D11_906_50_9_orig.png
│  │  │  ├── 2020-07-14_11-43-47_D11_906_50_9_inferred.wav
│  │  │  ├── 2020-07-14_11-43-47_D11_906_50_9_inferred.png
│  │  │  ├── 2020-07-14_11-43-47_D11_906_50_9_comparison.png
│  │  │  └── input.txt
│  │  ├── ...
│  ├── analysis
│  │  ├── 500_sims.log
│  │  ├── 500_3d.html
│  │  ├── 500_2d.html
│  │  ├── ...
│  ├── filelist
│  │  ├── weights.npy
│  │  ├── symbols.json
│  │  ├── filelist.csv
│  │  ├── audio_text_train_filelist.csv
│  │  ├── audio_text_test_filelist.csv
│  │  └── audio_text_val_filelist.csv
│  ├── checkpoints
│  │  ├── 0
│  │  ├── 500
│  │  ├── 1000
│  │  └── ...
│  ├── config.log
│  ├── weights_map.json
│  └── description.txt
│── training_...
```

# Maps

Maps can be created with `script_create_map_template.py`.

## Weight maps

These maps are used to map weights from a pretrained model to a new model where the two symbolsets differ.

## Inference maps

These maps are used to translate unknown symbols in the text which should be infered to symbols known to the model.

# Notes

## Requirements

- `numba==0.48` is needed because `librosa` otherwise fails later in runtime [see](https://github.com/librosa/librosa/issues/1160)
- `gdown` only required for downloading pretrained waveglow-model
- `wget` only required for automatically downloading datasets

## Configs

I also successfully tryed this configurations:
- Cuda 10.0, Nvidia driver 440.64.00, cuDNN 7.6.5 with GTX 1070 Mobile 8GB
- Cuda 10.2.89, Nvidia driver 440.100, cuDNN not installed with RTX 2070 8GB

to save requirements:
```bash
pipreqs .
```