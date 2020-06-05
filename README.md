# Tacotron 2

## Installation for Cuda 10.0, Nvidia driver 440.64.00, cuDNN 7.6.5 with GTX 1070 Mobile 8GB

```
./init.sh /datasets/models/taco2
```
examples:
north wind and the sun: from [wiki](https://en.wikipedia.org/wiki/The_North_Wind_and_the_Sun) and replace ɚ by ɹ̩ [see](https://en.wikipedia.org/wiki/R-colored_vowel)
narrow: ɾ do not exist
primary and secondary stress is not supported currently


```
$ git clone git@github.com:stefantaubert/tacotron2.git
$ cd tacotron2
$ git submodule init
$ git submodule update
$ sed -i -- 's,DUMMY,/datasets/LJSpeech-1.1/wavs,g' filelists/*.txt
$ conda create -n taco2pytorch python=3.6 -y
$ conda activate taco2pytorch
$ pip install -r req.txt
```

## FYI

apex is only required for fp16_run
```
git clone https://github.com/NVIDIA/apex
cd apex
conda activate taco2pytorch
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

with original requirements i get this error
ERROR: tensorflow 1.15.2 has requirement numpy<2.0,>=1.16.0, but you'll have numpy 1.13.3 which is incompatible.
ERROR: numba 0.49.1 has requirement numpy>=1.15, but you'll have numpy 1.13.3 which is incompatible.

theoretically pytorch for cuda 10.0 can be installed with (but works without it)
pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html


## Training
1. `python train.py`
2. (OPTIONAL) `./open-tensorboard.sh`

## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence  
By default, the dataset dependent text embedding layers are [ignored]

1. Download our published [Tacotron 2] model
2. `python train.py -c tacotron2_statedict.pt --warm_start`

## Multi-GPU (distributed) and Automatic Mixed Precision Training
1. `python -m multiproc train.py --hparams=distributed_run=True,fp16_run=True`

## Inference demo
`jupyter notebook --ip=127.0.0.1 --port=31337`
Load inference.ipynb

N.b.  When performing Mel-Spectrogram to Audio synthesis, make sure Tacotron 2
and the Mel decoder were trained on the same mel-spectrogram representation. 

## Notes
Size of:
- Val: 100
- Train: 12500
- Test: 500
Total (equals total possible): 13100
they took the first and third column out of metadata.csv

pretrain only contains 'state_dict'
- iteration, optimizer, learning_rate are not present

maybe create original text of LJSpeech and convert it to IPA and then split it again

training on pretrained model do not give results on first iterations