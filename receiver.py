"""
Programmer: Chris Tralie
Purpose: To extract simple pulse width modulated messages from audio
"""
import sounddevice as sd
from scipy.io.wavfile import write, read
from scipy.signal import butter, sosfiltfilt
import numpy as np
import argparse

def otsu(x):
    """
    Run Otsu's segmentation algorithm

    Parameters
    ----------
    x: ndarray(N)
        Samples on which to run Otsu's
    
    Returns
    -------
    score: float
        Score of best threshold
    thresh: float
        Best threshold
    """
    mn = int(np.min(x))
    mx = int(np.max(x))
    if mn == mx:
        return mn, np.inf
    threshs = np.arange(mn, mx)
    vals = np.zeros_like(threshs)
    for i, thresh in enumerate(threshs):
        vals[i] = np.nansum([np.mean(cls)*np.var(x, where=cls) for cls in [x >= thresh, x < thresh]])
    mn = np.min(vals)
    return mn, np.mean(threshs[vals == mn])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path at which to load audio, or 'mic' if recording")
    parser.add_argument('--seconds', type=int, default=30, help="Number of seconds to record")
    parser.add_argument("--fc", type=float, required=True, help="Carrier frequency")
    parser.add_argument('--sr', type=int, default=44100, help="Audio sample rate")
    parser.add_argument("--L", type=int, default=1000, help="Length of mini interval")
    parser.add_argument("--preamble_len", type=int, default=16, help="Length of preamble, in bits")
    opt = parser.parse_args()
    fc = opt.fc
    sr = opt.sr
    L = opt.L
    path = opt.path

    preamble = [1, 0]*(opt.preamble_len//2)
    code = ["100", "110"]
    preamble_expanded = "".join([code[b] for b in preamble])

    ## Step 1: Record audio and/or load it in
    if path == "mic":
        x = sd.rec(int(sr*opt.seconds), samplerate=sr, channels=1)
        sd.wait()
        x = x.flatten()
        x = x/np.max(np.abs(x))
        x = np.array(x*32767, dtype=np.int16)
        write("recorded.wav", sr, x)
        path = "recorded.wav"
    
    sry, y = read(path)
    assert(sry == sr)
    y = np.array(y, dtype=float)
    y = y/np.max(np.abs(y))

    ## Step 2: Filter out carrier frequency
    w1 = (fc*0.95)/sr
    w2 = (fc/0.95)/sr
    sos = butter(10, [w1*2, w2*2], 'bandpass', output='sos')
    y = sosfiltfilt(sos, y)

    ## Step 3: Extract bits from audio
    received = np.array([])
    best_score = np.inf
    t = np.arange(y.size)/sr
    c = np.cos(2*np.pi*fc*t)
    s = np.sin(2*np.pi*fc*t)
    for cutoff in range(0, L, L//50): # Find best shift based on success with Otsu's
        # Compute magnitude of this frequency in windows
        cm = c[cutoff:]*y[cutoff:]
        sm = s[cutoff:]*y[cutoff:]
        mag = cm**2 + sm**2
        mag = mag[0:L*(len(mag)//L)]
        mag = np.reshape(mag, (len(mag)//L, L))
        mag = np.sum(mag, axis=1)

        # Find the best threshold
        mn, thresh = otsu(mag)

        # Convert to binary and look for preamble
        receivedi = mag > thresh 
        receivedi = "".join(["%i"%x for x in receivedi])
        idx = None
        try:
            idx = receivedi.index(preamble_expanded)
        except:
            continue
        if mn < best_score:
            best_score = mn
            received = receivedi[idx+len(preamble_expanded):] # Cut off crap at beginning
    
    # Created an array of estimated bits based on patters of 100 (0), and 110 (1)
    received = np.array([int(x) for x in received])
    received = received[0:3*(len(received)//3)]
    received = np.reshape(received, (len(received)//3, 3))
    bits_est = np.array(np.sum(received, axis=1) > 1, dtype=int)

    # Group all groups of 8 bits together and convert to ascii
    bits_est = bits_est[0:8*(len(bits_est)//8)]
    bits_est = np.reshape(bits_est, (len(bits_est)//8, 8))
    cs = np.sum( [2**(7-i) for i in range(8)]*bits_est, axis=1)
    cs = [chr(c) for c in cs]
    print("--------------------")
    print("".join(cs))
    print("--------------------")