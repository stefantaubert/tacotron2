import torch
from src.common.audio.utils import wav_to_float32_tensor
from src.tacotron.layers import TacotronSTFT


class MelParser():
  def __init__(self, hparams):
    super().__init__()
    self.stft = TacotronSTFT(
      filter_length=hparams.filter_length,
      hop_length=hparams.hop_length,
      win_length=hparams.win_length,
      n_mel_channels=hparams.n_mel_channels,
      sampling_rate=hparams.sampling_rate,
      mel_fmin=hparams.mel_fmin,
      mel_fmax=hparams.mel_fmax
    )

  def get_mel_tensor_from_file(self, wav_path: str) -> torch.float32:
    wav_tensor, sr = wav_to_float32_tensor(wav_path)
    
    if sr != self.stft.sampling_rate :
      raise ValueError("{} {} SR doesn't match target {} SR".format(wav_path, sr, self.stft.sampling_rate))
    
    return self.get_mel_tensor(wav_tensor)

  def get_mel_tensor(self, wav_tensor: torch.float32) -> torch.float32:
    wav_tensor = wav_tensor.unsqueeze(0)
    wav_tensor = torch.autograd.Variable(wav_tensor, requires_grad=False)
    melspec = self.stft.mel_spectrogram(wav_tensor)
    melspec = melspec.squeeze(0)
    return melspec

if __name__ == "__main__":
  from src.tacotron.hparams import create_hparams
  from src.common.audio.utils import get_wav_tensor_segment
  wav_path = "/datasets/thchs_16bit_22050kHz_nosil/wav/train/A32/A32_11.wav"
  hparams = create_hparams()
  mel_parser = MelParser(hparams)
  mel = mel_parser.get_mel_tensor_from_file(wav_path)
  print(mel[:,:8])
  print(mel.size())
  
  wav_tensor, _ = wav_to_float32_tensor(wav_path)
  wav_tensor = get_wav_tensor_segment(wav_tensor, segment_length=4000)
  mel = mel_parser.get_mel_tensor(wav_tensor)
  print(mel[:,:8])
  print(mel.size())
