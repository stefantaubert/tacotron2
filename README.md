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

Serveraddress for example `user@example.com`.
Execute locally:
```bash
# generate ssh key
ssh-keygen -f ~/.ssh/abc-key-ecdsa -t ecdsa -b 521
# copy the public key to the server
ssh-copy-id -i ~/.ssh/abc-key-ecdsa user@example.com
# connect
ssh -i ~/.ssh/abc-key-ecdsa user@example.com
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

## Checkout repo

```bash
# get link from https://www.anaconda.com/products/individual and then 'Linux Python 3.7 64-Bit (x86) Installer' copy link
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
# reopen shell
rm Anaconda3-2020.02-Linux-x86_64.sh

git clone https://github.com/stefantaubert/tacotron2
cd tacotron2
git submodule init
git submodule update
conda create -n taco2pytorch python=3.6 -y
conda activate taco2pytorch
pip install -r requirements.txt
```
to be able to run training without being connected with ssh:
```
sudo apt install screen
```

check cuda is installed:
```
nvcc --version
```

error is normal:
```
ERROR: tensorflow 1.13.2 has requirement tensorboard<1.14.0,>=1.13.0, but you'll have tensorboard 2.2.2 which is incompatible.
Installing collected packages: numpy, pytz, six,
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
export base_dir="/home/stefan_taubert/taco2pt_v2"
export ljs_dir="/home/stefan_taubert/datasets/LJSpeech-1.1"
export ds_name="ljs_ipa"
python script_ljs_pre.py --base_dir=$base_dir --data_dir=$ljs_dir --ipa --ignore_arcs --ds_name=$ds_name --no_debugging
```

### Start training

duration: about 5 days on t4

```bash
export base_dir="/home/stefan_taubert/taco2pt_v2"
export hparams="batch_size=52,iters_per_checkpoint=500,epochs=300"
export speakers="ljs_ipa,1"
export custom_training_name="ljs_ipa_ms_from_scratch"
python paths.py --base_dir=$base_dir --custom_training_name=$custom_training_name --no_debugging
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --speakers=$speakers --hparams=$hparams --no_debugging
```

### Continue training

```bash
export base_dir="/home/stefan_taubert/taco2pt_v2"
export hparams="batch_size=52,iters_per_checkpoint=500,epochs=500"
export speakers="ljs_ipa,1"
export custom_training_name="ljs_ipa_ms_from_scratch"
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --hparams=$hparams --speakers=$speakers --continue_training --no_debugging
```

### Synthesize example

```bash
export base_dir="/home/stefan_taubert/taco2pt_v2"
export pretrained="/home/stefan_taubert/taco2pt_v2/pretrained"
python script_dl_waveglow_pretrained.py --pretrained_dir=$pretrained --no_debugging
export waveglow="/home/stefan_taubert/taco2pt_v2/pretrained/waveglow_256channels_universal_v5.pt"
export speakers="ljs_ipa,1"
export speaker="ljs_ipa,1"

export text="examples/ipa/north_sven_v2.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --no_debugging

export text="examples/en/north.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --no_debugging

export text="examples/en/democritus_v2.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --no_debugging
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
├── training_2020-06-17_11-11-03
│  ├── logs
│  │  └── log.txt
│  ├── inference
│  │  ├── 2020-06-18_11-11-03_democritus_A11_500
│  │  │  ├── config.log
│  │  │  ├── input.txt
│  │  │  ├── input_sentences.txt
│  │  │  ├── input_sentences_mapped.t
│  │  │  ├── input_symbols.txt
│  │  │  ├── input_map.json
│  │  │  └── 2020-06-17_18-11-03_democritus_A11_500.wav
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

I also successfylly tryed this configurations:
- Cuda 10.0, Nvidia driver 440.64.00, cuDNN 7.6.5 with GTX 1070 Mobile 8GB

to save requirements:
```bash
pipreqs .
```