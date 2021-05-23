import scipy.io.wavfile as wavfile 
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft,ifft

with open('laurel_yanny.wav', 'rb') as f:
    sampleRate, data = wavfile.read(f)

# b
fig, ax = plt.subplots()
ax.plot(data)
ax.set_xlabel('Time')
ax.set_ylabel('Strength')
plt.tight_layout()
plt.savefig('miniproj8.4.b.png')

# c
fig, ax = plt.subplots()
ax.plot(np.abs(fft(data)))
ax.set_xlabel('Frequency')
ax.set_ylabel('Strength')
plt.tight_layout()
plt.savefig('miniproj8.4.c.png')

# d
data_split = np.array_split(data,len(data)//500)
heatmap = np.zeros((80,len(data_split)))
for i in range(len(data_split)):
    split_fft = np.log(np.abs(fft(data_split[i])))
    for j in range(80):
        heatmap[j,i] = split_fft[j]
        
fig, ax = plt.subplots()
ax.imshow(heatmap, cmap='hot',origin='lower')
ax.set_xlabel('Chunk')
ax.set_ylabel('Fourier coefficient')
plt.tight_layout()
plt.savefig('miniproj8.4.d.png')

# e
data_fft = fft(data)
t_range = (1000,10000)
for t in np.linspace(*t_range):
    t = int(t)
    t_func_low, t_func_high = np.zeros(len(data)), np.zeros(len(data))
    t_func_low[:t] = 1
    t_func_low[-t:] = 1
    t_func_high[t:-t] = 1
    #t_func_high[-top:-t] = 1
    
    data_low = ifft(t_func_low*data_fft)
    data_high = ifft(t_func_high*data_fft)
    
    data_low = (data_low * 1.0 / np.max(np.abs(data_low)) * 32767).astype(np.int16)
    data_high = (data_high * 1.0 / np.max(np.abs(data_high)) * 32767).astype(np.int16)
    
    with open('high_{}.wav'.format(t),'wb') as out_high, open('low_{}.wav'.format(t),'wb') as out_low:
        wavfile.write(out_high,sampleRate,data_high)
        wavfile.write(out_low,sampleRate,data_low)
    
# f
data_ = (data * 1.0 / np.max(np.abs(data)) * 32767).astype(np.int16)
for r in [0.25,0.5,1,1.5,2]:
    with open('freq_{}.wav'.format(r),'wb') as out:
        wavfile.write(out,int(sampleRate*r),data_)
