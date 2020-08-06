# import torch
# import numpy as np
# from scipy.signal import get_window
# import librosa.util as librosa_util


# def griffin_lim(magnitudes, stft_fn, n_iters=30):
#   """
#   PARAMS
#   ------
#   magnitudes: spectrogram magnitudes
#   stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
#   """

#   angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
#   angles = angles.astype(np.float32)
#   angles = torch.autograd.Variable(torch.from_numpy(angles))
#   signal = stft_fn.inverse(magnitudes, angles).squeeze(1)

#   for i in range(n_iters):
#     _, angles = stft_fn.transform(signal)
#     signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
#   return signal

