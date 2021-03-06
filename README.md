# Tacotron 2 with IPA and Multispeaker Support

You can listen to sound samples [here](https://stefantaubert.github.io/tacotron2/)

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


## Setup

### Change Console Encoding to display IPA chars

Check with `locale` which setting you have and if `en_US.UTF-8` is not in the values of `LANG` or `LC_*` then change them:

```sh
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
# note: execute before before you open a screen
```

and ensure you have an updated screen:

```sh
sudo apt upgrade screen
```

### Locally with remote Server

Serveraddress for example `joedoe@example.com`.

#### SSH login

Execute locally:

```bash
# generate ssh key
ssh-keygen -f ~/.ssh/abc-key-ecdsa -t ecdsa -b 521
# copy the public key to the server
ssh-copy-id -i ~/.ssh/abc-key-ecdsa joedoe@example.com
# connect
ssh -i ~/.ssh/abc-key-ecdsa joedoe@example.com
```

#### samba access to get synthesized files

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

### Create Google Cloud Platform VM (optional)

#### Prerequisites

You need to upgrade your free account to a paid account (you retain your free money).
For this training you approximately need 100$.
You also need to inclease your GPU quota at least for the T4 GPU to one.
You can also use an other GPU if you want, the code is optimized for a GPU with 16GB ram.

#### Create VM

Create VM with pytorch 1.4 and Cuda 10.1 already installed and configured using *Google Cloud Shell*:

```bash
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

### Checkout repo

```bash
git clone https://github.com/stefantaubert/tacotron2
cd tacotron2
sudo apt install -y python3.8 python3.8-distutils python3.8-dev
python3.8 -m pip install pipenv
python3.8 -m pipenv install --dev
```

to be able to run training without being connected via ssh:

```bash
sudo apt install screen
# usage: screen
```

check cuda is installed:

```bash
nvcc --version
```

### IPA synthesis using LJSpeech-1.1 dataset

#### Install flite

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

#### Automatically download and prepare dataset

duration: about 1.5h

```bash
```

#### Start training

duration: about 5 days on t4

```bash
```

#### Continue training

```bash
```

#### Synthesize example

```bash
```

### Filestructure

```txt
$base_dir

├── pre
│  ├── ds
│  │  ├── ljs
│  │  │  ├── speakers.json
│  │  │  ├── symbols.json
│  │  │  ├── accents.json
│  │  │  ├── data.csv
│  │  │  ├── text
│  │  │  │  ├── en
│  │  │  │  │  ├── data.csv
│  │  │  │  │  ├── symbol_ids.json
│  │  │  │  │  ├── symbols.json
│  │  │  │  │  ├── accents.json
│  │  │  │  ├── en_normalized
│  │  │  │  │  ├── data.csv
│  │  │  │  │  ├── symbol_ids.json
│  │  │  │  │  ├── symbols.json
│  │  │  │  │  ├── accents.json
│  │  │  │  ├── ipa_normalized
│  │  │  │  │  ├── data.csv
│  │  │  │  │  ├── symbol_ids.json
│  │  │  │  │  ├── symbols.json
│  │  │  │  │  ├── accents.json
│  │  │  ├── wav
│  │  │  │  ├── 22050kHz
│  │  │  │  │  ├── data.csv
│  │  │  │  │  ├── 0-499
│  │  │  │  │  │  ├── 0.wav
│  │  │  │  │  │  ├── ...
│  │  │  │  │  ├── ...
│  │  │  ├── mel
│  │  │  │  ├── 22050kHz
│  │  │  │  │  ├── data.csv
│  │  │  │  │  ├── 0-499
│  │  │  │  │  │  ├── 0.pt
│  │  │  │  │  │  ├── ...
│  │  │  │  │  ├── ...
│  │  ├── thchs
│  │  │  ├── data.csv
│  │  │  ├── speakers.json
│  │  │  ├── text
│  │  │  │  ├── ipa
│  │  │  │  │  ├── data.csv
│  │  │  │  │  ├── symbols.json
│  │  │  ├── wav
│  │  │  │  ├── 16000kHz
│  │  │  │  │  ├── data.csv
│  │  │  │  ├── 16000kHz_normalized
│  │  │  │  │  ├── data.csv
│  │  │  │  │  ├── 0-499
│  │  │  │  │  │  ├── 0.wav
│  │  │  │  │  │  ├── ...
│  │  │  │  ├── 22050kHz_normalized
│  │  │  │  │  ├── data.csv
│  │  │  │  │  ├── 0-499
│  │  │  │  │  │  ├── 0.wav
│  │  │  │  │  │  ├── ...
│  │  │  │  │  ├── trim
│  │  │  │  │  │  ├── 499
│  │  │  │  │  │  │  ├── original.wav
│  │  │  │  │  │  │  ├── original.png
│  │  │  │  │  │  │  ├── cs=5,ts=-20dBFS,bs=100ms,te=-30dBFS,be=150ms.wav
│  │  │  │  │  │  │  ├── cs=5,ts=-20dBFS,bs=100ms,te=-30dBFS,be=150ms.png
│  │  │  │  ├── 22050kHz_normalized_nosil
│  │  │  │  │  ├── data.csv
│  │  │  │  │  ├── 0-499
│  │  │  ├── mel
│  │  │  │  ├── 22050kHz_normalized_nosil
│  │  │  │  │  ├── data.csv
│  │  │  │  │  ├── 0-499
│  ├── prepared
│  │  ├── thchs
│  │  │  ├── speakers.json
│  │  │  ├── symbols.json
│  │  │  ├── data.csv
│  │  │  ├── north
│  │  │  │  ├── text.csv
│  │  │  │  ├── accents.csv
│  │  │  │  ├── inference.csv
│  │  │  │  ├── symbols.json
│  │  ├── ljs
│  │  ├── ljs_thchs
├── pre
│  ├── ds
│  │  ├── thchs
│  │  │  ├── speakers.json
│  │  │  ├── data.csv
│  │  ├── ljs
│  │  │  ├── speakers.json
│  │  │  ├── data.csv
│  ├── wav
│  │  ├── ljs_22050kHz
│  │  │  ├── data.csv
│  │  ├── thchs_16000kHz
│  │  │  ├── data.csv
│  │  ├── thchs_16000kHz_normalized
│  │  │  ├── data.csv
│  │  │  ├── data
│  │  │  │  ├── 0-LJ001-0001.wav
│  │  │  │  ├── ...
│  │  ├── thchs_22050kHz_normalized
│  │  │  ├── data.csv
│  │  │  ├── data
│  │  │  │  ├── 0-LJ001-0001.wav
│  │  │  │  ├── ...
│  │  ├── thchs_22050kHz_normalized_nosil
│  │  │  ├── data.csv
│  │  │  ├── data
│  │  │  │  ├── 0-LJ001-0001.wav
│  │  │  │  ├── ...
│  ├── text
│  │  ├── ljs_en
│  │  │  ├── data.csv
│  │  │  ├── symbols.json
│  │  ├── ljs_ipa
│  │  │  ├── data.csv
│  │  │  ├── symbols.json
│  │  ├── thchs_ipa
│  │  │  ├── data.csv
│  │  │  ├── symbols.json
│  ├── mel
│  │  ├── ljs_22050kHz
│  │  │  ├── data.csv
│  │  │  ├── data
│  │  │  │  ├── 0-LJ001-0001.pt
│  │  │  │  ├── ...
│  │  ├── thchs_22050kHz_normalized_nosil
│  │  │  ├── data.csv
│  │  │  ├── data
│  │  │  │  ├── 0-LJ001-0001.pt
│  │  │  │  ├── ...
├── waveglow
│  ├── training1
│  ├── ...
├── tacotron
│  ├── training1
│  │  ├── settings.json
│  │  ├── train.csv
│  │  ├── test.csv
│  │  └── validation.csv
│  │  ├── logs
│  │  │  └── log.txt
│  │  ├── inference
│  │  │  ├── 2020-09-16,16-46-41,text=ipa-north_sven_orig,speaker=0,it=29380
│  │  │  │  ├── 2020-09-16,16-46-41,text=ipa-north_sven_orig,speaker=0,it=29380.txt
│  │  │  │  ├── 2020-09-16,16-46-41,text=ipa-north_sven_orig,speaker=0,it=29380_v.png
│  │  │  │  ├── 2020-09-16,16-46-41,text=ipa-north_sven_orig,speaker=0,it=29380_h.png
│  │  │  │  ├── 2020-09-16,16-46-41,text=ipa-north_sven_orig,speaker=0,it=29380_v_pre.png
│  │  │  │  ├── 2020-09-16,16-46-41,text=ipa-north_sven_orig,speaker=0,it=29380_v_alignments.png
│  │  │  │  ├── 2020-09-16,16-46-41,text=ipa-north_sven_orig,speaker=0,it=29380.wav
│  │  │  │  ├── 1_v_pre.png
│  │  │  │  ├── 1_v_alignments.png
│  │  │  │  ├── 1.wav
│  │  │  │  ├── ...
│  │  │  ├── ...
│  │  ├── validation
│  │  │  ├── 2020-07-14_11-43-47_D11_906_50_9
│  │  │  │  ├── 2020-07-14_11-43-47_D11_906_50_9.txt
│  │  │  │  ├── 2020-07-14_11-43-47_D11_906_50_9_orig.wav
│  │  │  │  ├── 2020-07-14_11-43-47_D11_906_50_9_orig.png
│  │  │  │  ├── 2020-07-14_11-43-47_D11_906_50_9_inferred.wav
│  │  │  │  ├── 2020-07-14_11-43-47_D11_906_50_9_inferred.png
│  │  │  │  ├── 2020-07-14_11-43-47_D11_906_50_9_comparison.png
│  │  │  ├── ...
│  │  ├── analysis
│  │  │  ├── 500.txt
│  │  │  ├── 500_2d.html
│  │  │  ├── 500_3d.html
│  │  │  ├── ...
│  │  ├── checkpoints
│  │  │  ├── 0.pt
│  │  │  ├── 500.pt
│  │  │  ├── 1000.pt
│  │  │  └── ...
│  │── ...
```

## Maps

Maps can be created with `create_map_template.py`.

### Weight maps

These maps are used to map weights from a pretrained model to a new model where the two symbolsets differ.

### Inference maps

These maps are used to translate unknown symbols in the text which should be infered to symbols known to the model.

## Notes

Create custom pylint file: `pylint --generate-rcfile > pylintrc`
Sort imports on save:

```json
{
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  }
}
```

For running isort on all files (execute in workspace dir): `isort . -rc`

Following setting to prevent that autoimports do not include "src." prefix [see](https://github.com/microsoft/pylance-release):

```json
{
  "python.analysis.autoSearchPaths": false,
}
```

Install Praat:

```sh
sudo apt-get install praat
```

### Requirements

- `gdown` required for downloading pretrained waveglow-model
- `wget` required for automatically downloading datasets
- `scikit-image` required for comparing mels for waveglow evaluation

### Configs

I also successfully tryed this configurations:

- Cuda 10.0, Nvidia driver 440.64.00, cuDNN 7.6.5 with GTX 1070 Mobile 8GB
- Cuda 10.2.89, Nvidia driver 440.100, cuDNN not installed with RTX 2070 8GB

## How to prepare texts

- "intrinsically blue," -> "intrinsically blue",
- "intrinsically blue." -> "intrinsically blue".
- "5cm" -> "5 cm"
- "3.5*10^-5 -> "3.5e-5"
