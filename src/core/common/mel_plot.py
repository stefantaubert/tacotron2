import imageio
import matplotlib.pylab as plt
import numpy as np
from librosa.filters import mel as librosa_mel_fn
from skimage.metrics import structural_similarity

def compare_mels(path_a, path_b):
  img_a = imageio.imread(path_a)
  img_b = imageio.imread(path_b)
  #img_b = imageio.imread(path_original_plot)
  assert img_a.shape[0] == img_b.shape[0]
  img_a_width = img_a.shape[1]
  img_b_width = img_b.shape[1]
  resize_width = img_a_width if img_a_width < img_b_width else img_b_width
  img_a = img_a[:,:resize_width]
  img_b = img_b[:,:resize_width]
  #imageio.imsave("/tmp/a.png", img_a)
  #imageio.imsave("/tmp/b.png", img_b)
  score, diff_img = structural_similarity(img_a, img_b, full=True, multichannel=True)
  #imageio.imsave(path_out, diff)
  return score, diff_img

def plot_melspec(mel, mel_dim_x=16, mel_dim_y=5, factor=1, title=None):
  height, width = mel.shape
  width_factor = width / 1000
  _, axes = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(mel_dim_x*factor*width_factor, mel_dim_y*factor),
  )

  axes.set_title(title)
  axes.set_yticks(np.arange(0, height, step=10))
  axes.set_xticks(np.arange(0, width, step=100))
  axes.set_xlabel("Samples")
  axes.set_ylabel("Freq. channel")
  axes.imshow(mel, aspect='auto', origin='bottom', interpolation='none')
  return axes
  

if __name__ == "__main__":
  pass
  # from src.tacotron.hparams import create_hparams
  # from src.core.common import get_wav_tensor_segment
  # wav_path = "/datasets/thchs_16bit_22050kHz_nosil/wav/train/A32/A32_11.wav"
  # hparams = create_hparams()
  # mel_parser = MelParser(hparams)
  # mel = mel_parser.get_mel_tensor_from_file(wav_path)
  # print(mel[:,:8])
  # print(mel.size())

  # wav_tensor, _ = wav_to_float32_tensor(wav_path)
  # wav_tensor = get_wav_tensor_segment(wav_tensor, segment_length=4000)
  # mel = mel_parser.get_mel_tensor(wav_tensor)
  # print(mel[:,:8])
  # print(mel.size())
