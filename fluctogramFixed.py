"""
Extract Fluctogram feature from audio
"""

import essentia
import essentia.standard
from essentia.standard import *
import numpy as np
from scipy.ndimage.interpolation import shift

FRAMESIZE = 4096
HOPSIZE = 1024

[SAMPLERATE,audio] = scipy.io.wavfile.read('sample.wav')

if len(audio.shape) == 2:
	audio = audio[:,0]/2 + audio[:,1]/2

audio = essentia.array(audio)

w = Windowing(type = 'hann', size=FRAMESIZE)
spectrum = Spectrum()
spectrogram = []

for frame in FrameGenerator(audio, frameSize=FRAMESIZE, hopSize=HOPSIZE):
    dft = spectrum(w(frame))
    spectrogram.append(dft)

spectrogram = essentia.array(spectrogram)

# Map the spectrum into pitch scale
pitchScale = []    # Value is spectrum index

for i in range(120*6+1):    # 10 bins per semitone. 6 octave.
    freq = 164.814 * 2 ** (i / 120.0)    # Lowest note is E3 (164.814 Hz)
    idx = int(np.round(freq / SAMPLERATE * (FRAMESIZE / 2)))
    pitchScale.append(idx)


maxidx = np.zeros([len(spectrogram),17])

# 240 bins per band. 17 bands.
for i in range(17):
    freqRange = [pitchScale[i * 30], pitchScale[i * 30 + 240]]
    # Weight each band by triangle window
    weightedSpectrogram = spectrogram
    bandwidth = freqRange[1] - freqRange[0]
    wTri = np.empty([bandwidth, ])
    for j in range(bandwidth):
        wTri[j] = 1 - abs(2.0 / (bandwidth - 1) * ((bandwidth - 1) / 2.0 - j))
    for k in range(len(spectrogram)):
	weightedSpectrogram[k, freqRange[0] : freqRange[1]] = wTri * weightedSpectrogram[k, freqRange[0] : freqRange[1]]
	if (k>0):
	    for n in range(-5,5):
	        correlation = np.correlate(weightedSpectrogram[k-1,freqRange[0] : freqRange[1]],shift(weightedSpectrogram[k,freqRange[0] : freqRange[1]],n,cval=0))
		if (correlation > maxidx[k-1,i]):
		    maxidx[k-1,i] = n
		



"""
My edited part 
"""
Flucts = weightedSpectrogram[:,0:17]
Fluct = numpy.empty([len(weightedSpectrogram), 17])
for i in range(len(weightedSpectrogram)):
    for j in range(17):
        # variance of first five MFCCs (excluding the 0th) over 11 successive frames centered on the current frame
        Fluct[i][j] = numpy.var(weightedSpectrogram[max(0, i-20) : min(len(weightedSpectrogram), i+20), j])


