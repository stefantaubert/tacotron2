def get_basename(values: tuple):
  wav = values[0]
  return wav

def get_speaker_name(values: tuple):
  wav = values[1]
  return wav

def get_text(values: tuple):
  wav = values[2]
  return wav

def get_path(values: tuple):
  wav = values[3]
  return wav

def to_values(name, speaker_name, text, wav_path):
  return (name, speaker_name, text, wav_path)
