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
git clone git@github.com:stefantaubert/tacotron2.git
cd tacotron2
git submodule init
git submodule update
conda create -n taco2pytorch python=3.6 -y
conda activate taco2pytorch
pip install -r reqmin.txt
# TODO: reqmin is not enough
pip install -r reqmax.txt
```

## IPA synthesis using LJSpeech-1.1 dataset

### Install flite

If you want to train on IPA-Symbols you need to install [flite](https://github.com/festvox/flite) for G2P conversion of English text:

```bash
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
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --is_ipa --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --no_debugging

export text="examples/en/north.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --no_debugging

export text="examples/en/democritus_v2.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --no_debugging
```

# Notes

I also successfylly tryed this configurations:
- Cuda 10.0, Nvidia driver 440.64.00, cuDNN 7.6.5 with GTX 1070 Mobile 8GB

