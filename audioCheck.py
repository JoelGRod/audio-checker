import pathlib

import numpy
from pydub import AudioSegment
from scipy import signal
from mutagen.mp3 import MP3

def check(path = 'mp3', **kwargs):
    path = pathlib.Path(path)
    if(path.is_file()):
        print(check_file(path, **kwargs))
        return
    if(path.is_dir()):
        for p in path.glob("**/*"):
            if p.suffix in [".mp3", ".wav", ".flac"]:
                print(check_file(p, **kwargs))

def check_file(filename, window_length_s: float = 0.05, channel: int = 0):
    track = AudioSegment.from_file(filename)

    assert track.channels is not None
    out = numpy.array(track.get_array_of_samples()).reshape(-1, track.channels)

    if window_length_s is None:
        nperseg = None
    else:
        nperseg = int(round(window_length_s * track.frame_rate))

    # Use the first channel by default
    f, _, Sxx = signal.spectrogram(
        out[:, channel],
        fs=track.frame_rate,
        scaling="spectrum",
        mode="magnitude",
        # nperseg=window_length_samples
        nperseg=nperseg,
        # noverlap=noverlap
    )
    # Make sure all values are positive for the log scaling
    smallest_positive = numpy.min(Sxx[Sxx > 0])
    Sxx[Sxx < smallest_positive] = smallest_positive
    # Which row surpasses the average first?
    log_Sxx = numpy.log10(Sxx)
    avg_log_Sxx = numpy.average(log_Sxx)
    count = numpy.sum(log_Sxx > avg_log_Sxx, axis=1)
    k = numpy.where(count > log_Sxx.shape[1] / 8)[0][-1]

    # What do we expect?
    filename = pathlib.Path(filename)
    if filename.suffix in [".wav", ".flac"]:
        if f[k] > 19000:
            return f"{filename} seems good."
        else:
            return f"{filename} is WAV, but has max frequency about {f[k]:.0f} Hz."
    elif filename.suffix == ".mp3":
        mp3_file = MP3(filename)
        bitrate = int(mp3_file.info.bitrate / 1000)
        bitrate_to_min_max_freq = {
            0: 0,
            16: 4000,
            64: 10000,
            128: 15000,
            192: 16000,
            320: 18000,
        }

        for key, val in bitrate_to_min_max_freq.items():
            if bitrate < key:
                break
            expected_max_freq = val

        if f[k] > expected_max_freq:
            return f"{filename} seems good [{bitrate} kbps]."
        else:
            return f"{filename} is MP3 [{bitrate} kbps], but has max frequency about {f[k]:.0f} Hz."
            
    else:
        return f"Don't know what to expect for {filename}."


