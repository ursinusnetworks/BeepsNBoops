"""
Programmer: Chris Tralie
Purpose: To create simple pulse width modulated outputs with different
carrier frequencies as envelopes
"""
from scipy.io.wavfile import write
import numpy as np
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages", type=str, required=True, help="Path to JSON file containing messages")
    parser.add_argument("--path", type=str, required=True, help="Path to which to save audio")
    parser.add_argument('--sr', type=int, default=44100, help="Audio sample rate")
    parser.add_argument("--L", type=int, default=1000, help="Length of mini interval")
    parser.add_argument("--preamble_len", type=int, default=16, help="Length of preamble, in bits")
    opt = parser.parse_args()
    sr = opt.sr
    L = opt.L
    preamble = [1, 0]*(opt.preamble_len//2)

    messages = json.load(open(opt.messages))
    ys = []
    for m in messages:
        ## Step 1: Create the message
        msg = m["message"]
        msg = [format(ord(c), "08b") for c in msg]
        msg = "".join(msg)
        msg = [int(c) for c in msg]
        msg = preamble + msg

        ## Step 2: Create the envelope
        env = np.zeros(len(msg)*L*3)
        for i, bit in enumerate(msg):
            env[L*i*3:L*(i*3+1)] = 1
            if bit == 1:
                env[L*(i*3+1):L*(i*3+2)] = 1
        
        ## Step 3: Create a carrier modulated by the envelope
        fc = m["fc"]
        t = np.arange(env.size)/sr
        y = np.cos(2*np.pi*fc*t)*env
        ys.append(y)
    
    N = max([y.size for y in ys])
    y = np.zeros(N)
    for yi in ys:
        y[0:yi.size] += yi

    y = y/np.max(np.abs(y))
    y = np.array(y*32767, dtype=np.int16)
    write(opt.path, sr, y)