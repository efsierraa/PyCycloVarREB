# %% Loading
import os # change dir
Wdir  = os.getcwd()
Wdir2 = Wdir
import sys 
sys.path.append(Wdir+'\functions')# add to path
import numpy as np # np matrix functions
import matplotlib.pyplot as plt # plotting pack
from matplotlib.gridspec import GridSpec # easy subplots AND BEAUTIFUL FIGS
import scipy.signal as sg # signal pack
import scipy.io as sio # load save matlab files
from functions.REB_functions import bearingfault,noise_snr,tTacho_fsig,COT_intp,\
SK_W,PSD_envW,printxtips
plt.close('all')
# %% Numerical Signal
plt.rc('font', size=14) # default text size
# geometry bearing 16002
#type : 0:'BPFO' pour défaut de bague externe
#       1:'BPFI' pour défaut de bague interne
dsurD = 15/32;N = 8;fo = 0;fi = [5,60];typeF = 1;fr = 5e3;fs = 2**14;T  = 10;
data,TPT=bearingfault(dsurD, N, fo,fi,typeF,fr,fs,T)

f0e = data[:,0]
f0 = sg.hilbert(f0e)
f0 = np.diff(np.unwrap(np.angle(f0)))*fs/TPT/(2*np.pi)
f0 = np.r_[f0,f0[-1]]
wfiltsv = int(2**np.log2(int(.05*fs))-1)
f0= sg.savgol_filter(f0,wfiltsv,2)# smoothing filter

data = data[:,1]
data = noise_snr(data,3,'white')
t = np.r_[:len(data)]/fs
# %% SK filtering baseline 00
Nw       = 2**np.fix(np.log2(fs*1/2))*2**3  # 1/2 minimun expected frq 2Hz
Nw       = int(Nw)
Nfft     = 2*Nw
#Noverlap = round(0.945*Nw)
Noverlap = round(3/4*Nw)

SK,_,_,_ = SK_W(data,Nfft,Noverlap,Nw)
b        = np.fft.fftshift(np.real(np.fft.ifft(SK)))
datas   = sg.fftconvolve(data,b,mode='same')
#datas0   = (datas0-np.mean(datas0))/np.std(datas0) # normalization

# % angular resampling et envelope spectrum
isencod=1
_,tTacho,_  = tTacho_fsig(f0e,fs,TPT,isencod)
SampPerRev = 2**np.fix(np.log2(fs/2/fi[1]))*2*2
datas0,tc,SampPerRev = COT_intp(datas,tTacho,TPT,fs,SampPerRev)

datae0 = sg.hilbert(datas0)
datae0 = np.abs(datas0)**2
datae0 = datae0-np.mean(datae0)
sp = np.abs(np.fft.fft(datae0)/len(datae0))**2
freq = np.fft.fftfreq(datae0.shape[-1])

# plot spectrum
fig = plt.figure()
sp = sp[:int(len(sp)/2)]
freq = freq[:int(len(freq)/2)]
freq = freq*SampPerRev
plt.plot(freq, sp)
plt.axis([freq[0],100,0,1.1*max(sp)])
plt.xlabel('Order [xn]')
plt.ylabel('PSD')
plt.legend([r"$\left|X(\Theta)\right|^2$"])

printxtips(freq,sp,xheigh0=5.875,deltapsd=1/2,K=16)
plt.show()
plt.savefig(Wdir2+'/Figures/exp04_envelopeSP_baseline_RB.pdf',bbox_inches='tight')
# %% Angular resampling & envelope spectrum Baseline 01
isencod=1
_,tTacho,_  = tTacho_fsig(f0e,fs,TPT,isencod)
SampPerRev = 2**np.fix(np.log2(fs/2/fi[1]))*2*2
datas,tc,SampPerRev = COT_intp(data,tTacho,TPT,fs,SampPerRev)

# SK filtering
Nw       = 2**np.fix(np.log2(SampPerRev*1/2))*2**3  # 1/2 minimun expected frq 2Hz
Nw       = int(Nw)
Nfft     = 2*Nw
#Noverlap = round(0.945*Nw)
Noverlap = round(3/4*Nw)

SK,_,_,_ = SK_W(datas,Nfft,Noverlap,Nw)
b        = np.fft.fftshift(np.real(np.fft.ifft(SK)))
datas    = sg.fftconvolve(datas,b,mode='same')

# envelope spectrum
datae = sg.hilbert(datas)
datae = np.abs(datas)**2
datae = datae-np.mean(datae)

# % plot envelope spectrum
fig = plt.figure()
sp = np.abs(np.fft.fft(datae)/len(datae))**2
sp = sp[:int(len(sp)/2)]
freq = np.fft.fftfreq(datae.shape[-1])
freq = freq[:int(len(freq)/2)]
freq = freq*SampPerRev
plt.plot(freq, sp)
plt.axis([freq[0],100,0,1.1*max(sp[(freq>2)*(freq<100)])])
plt.xlabel('Order [xn]')
plt.ylabel('PSD')
plt.legend([r"$\left|X(\Theta)\right|^2$"])

printxtips(freq,sp,xheigh0=5.875,deltapsd=1/2,K=16)
plt.show()
plt.savefig(Wdir2+'/Figures/exp04_envelopeSP.pdf',bbox_inches='tight')#, bbox_inches="tight"
# %% Angular resampling & envelope spectrum Proposal
isencod=1
_,tTacho,_  = tTacho_fsig(f0e,fs,TPT,isencod)
SampPerRev = 2**np.fix(np.log2(fs/2/fi[1]))*2*2
datas,tc,SampPerRev = COT_intp(data,tTacho,TPT,fs,SampPerRev)

#% SK filtering proposal
Nw1      = 2**int(np.log2(1/2*SampPerRev))*2**(5+3) # 2 Hz minimum failure exp
Nw2      = 2**int(np.log2(1/2*SampPerRev))*2**3
Nfft     = 2*Nw1
Nfft2    = 2*Nw2
# 7.667 evp failure found 7.759 evp theoretical
Noverlap = round(.95*Nw1)#68-95-s997
Window   = np.kaiser(Nw1,beta=0) # beta 0 rectangular,5	Similar to a Hamming
# 6	Similar to a Hanning, 8.6	Similar to a Blackman

#Window   = np.ones(Nw1) # First window for SK per segments
filterr  = 1 # filtering with SK
psd,f,K  = PSD_envW(datas,Nfft,Noverlap,Window,Nw2,Nfft2,filterr)
f        = f*SampPerRev

# % plot envelope spectrum
fig = plt.figure()
plt.plot(f, psd)
plt.xlabel('Order [xn]')
plt.ylabel('PSD')
plt.axis([f[0],100,0,1.1*max(psd[f>.5])])
plt.legend([r"$\left|X(\Theta)\right|^2$"])

printxtips(f,psd,xheigh0=5.875,deltapsd=1/2,K=16)
plt.show()
plt.savefig(Wdir2+'/Figures/exp04_envelopeSP_proposal.pdf',bbox_inches='tight')
#%% plotting data
fig = plt.figure()
plt.plot(t,data)
plt.xlim(t[0],t[fs])
plt.xlabel('Time [sec]')
plt.show()
plt.savefig(Wdir2+'/Figures/exp04_numsign.pdf',bbox_inches='tight')
#%% plotting IF profile
fig = plt.figure()
plt.plot(t,f0)
plt.xlim(t[0],t[-1])
plt.xlabel('Time [sec]')
plt.ylabel('Frequency [Hz]')
plt.show()
plt.savefig(Wdir2+'/Figures/exp04_IFprofile.pdf',bbox_inches='tight')
#%% plotting original PSD
fig = plt.figure()
sp = np.abs(np.fft.fft(data)/len(data))**2
sp = sp[:int(len(sp)/2)]
freq = np.fft.fftfreq(t.shape[-1])
freq = freq[:int(len(freq)/2)]
freq = freq*fs
plt.plot(freq, sp)
plt.axis([freq[0],freq[-1],0,1.1*max(sp)])
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD')
plt.legend([r"$\left|X(w)\right|^2$"])
plt.show()
plt.savefig(Wdir2+'/Figures/exp04_psd.pdf',bbox_inches='tight')

## spectrogram
#Pxx, freqs, bins, im = ax2.specgram(data_r, NFFT=NFFT, Fs=fsr, noverlap=NFFT/2,cmap='viridis')
#ax2.plot(t2,K*f_alpha_est[:,2*n[-1]],t2,f_alpha_est[:,2*n[-1]],)
#ax2.legend(['$f_{6}[n]$','$f_{1}[n]$'])
#ax2.set_xlim([t[0],t[-1]])
#ax2.set_ylim([0,fsr/2])
#ax2.set_ylabel('Frequency [Hz]')
#ax2.set_xlabel('Time [sec]')
