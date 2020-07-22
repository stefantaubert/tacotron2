from pydub import AudioSegment


def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10, buffer_ms: int = 0):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    assert chunk_size > 0

    trim_ms = 0

    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
      trim_ms += chunk_size

    if buffer_ms <= trim_ms:
      trim_ms -= buffer_ms

    return trim_ms

def remove_silence(
    in_path: str,
    out_path: str,
    chunk_size: int,
    threshold_start: float,
    threshold_end: float,
    buffer_start_ms: float,
    buffer_end_ms: float
  ):

  sound = AudioSegment.from_file(in_path, format="wav")

  start_trim = detect_leading_silence(
    sound=sound,
    silence_threshold=threshold_start,
    chunk_size=chunk_size,
    buffer_ms=buffer_start_ms
  )

  end_trim = detect_leading_silence(
    sound=sound.reverse(),
    silence_threshold=threshold_end,
    chunk_size=chunk_size,
    buffer_ms=buffer_end_ms
  )
  
  duration = len(sound)
  trimmed_sound = sound[start_trim:duration - end_trim]
  trimmed_sound.export(out_path, format="wav")
