import numpy as np # math pack!
import scipy.signal as sg # DSP
from numpy import linalg as LA # Linear algebra
import sys # sio packagesudo

# %% Spectral Kurtosis
def SK_W(x,Nfft,Noverlap,Window):
# [SK,M4,M2,f] = SK_W(x,Nfft,Noverlap,Window) 
# Welch's estimate of the spectral kurtosis       
#       SK(f) = M4(f)/M2(f)^2 - 2 
# where M4(f) = E{|X(f)|^4} and M2(f) = E{|X(f)|^2} are the fourth and
# second order moment spectra of signal x, respectively.
# Signal x is divided into overlapping blocks (Noverlap taps), each of which is
# detrended, windowed and zero-padded to length Nfft. Input arguments nfft, Noverlap, and Window
# are as in function 'PSD' or 'PWELCH' of Matlab. Denoting by Nwind the window length, it is recommended to use 
# nfft = 2*NWind and Noverlap = 3/4*Nwind with a hanning window.
# (note that, in the definition of the spectral kurtosis, 2 is subtracted instead of 3 because Fourier coefficients
# are complex circular)
#
# --------------------------
# References: 
# J. Antoni, The spectral kurtosis: a useful tool for characterising nonstationary signals, Mechanical Systems and Signal Processing, Volume 20, Issue 2, 2006, pp.282-307.
# J. Antoni, R. B. Randall, The spectral kurtosis: application to the vibratory surveillance and diagnostics of rotating machines, Mechanical Systems and Signal Processing, Volume 20, Issue 2, 2006, pp.308-331.
# --------------------------
# Author: J. Antoni
# Last Revision: 12-2014
# --------------------------
    if type(Window) == int:
        Window = np.hanning(Window)
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.hanning.html
    Window = Window.reshape(-1,1)[:,0]/LA.norm(Window)	# window normalization
    n = len(x)  							        # number of data points
    nwind = len(Window)     				        # length of window
    
    if nwind <= Noverlap:
        print('Window length must be > Noverlap')
        input()
        sys.exit(1)
    if Nfft < nwind:
        print('Window length must be <= Nfft')
        input()        
        sys.exit(1)
    
    x = x.reshape(-1,1)[:,0]	
    k = np.fix((n-Noverlap)/(nwind-Noverlap)) # number of windows
    
    
    # Moment-based spectra
    index = np.r_[:nwind]
    f = np.r_[:Nfft]/Nfft;
    M4 = 0;
    M2 = 0;
    
    for i in np.r_[:k]:
        xw = Window*x[index]
        Xw = np.fft.fft(xw,n=Nfft)/Nfft        
        M4 = np.abs(Xw)**4 + M4
        M2 = np.abs(Xw)**2 + M2
        index = index + (nwind - Noverlap)
    
    # normalize
    M4 = M4/k;   
    M2 = M2/k; 
    
    # spectral kurtosis 
    SK = M4/M2**2 - 2
    
    # reduce bias near f = 0 mod(1/2)
    W = np.abs(np.fft.fft(Window**2,Nfft))**2
    Wb = np.zeros([Nfft,1])[:,0]
    for i in np.r_[:Nfft]:
       Wb[i] = W[np.mod(2*i,Nfft)]/W[0]
       
    SK = SK - Wb
    
    return SK,M4,M2,f

# % PSD filtered with sliding Spectral Kurtosis
def PSD_envW(x,nfft,Noverlap,Window,Nwind2,nfft2,filterr):
# variables x,nfft,Noverlap,Window are the same as in SK_W
# Nwind2 is the length of the sliding window it must be larger than Window
# nfft2 is the resolution of the filtered PSD
# filterr is the filter is a boolean variable if true the PSD is filtered with SK_W in a sliding basis
# --------------------------------------------------
# Author: Edgar F. Sierra-Alonso
# Last Revision: 02-2022
# --------------------------------------------------
    Window = Window.reshape(-1,1)[:,0]
    n = len(x)		# Number of data points
    nwind = len(Window) # length of window
    x = x.reshape(-1,1)[:,0]
    K = int(np.fix((n-Noverlap)/(nwind-Noverlap))) # Number of windows
    Noverlap2 = int(np.round(3/4*Nwind2))
    Window2 = np.hanning(Nwind2)
    
    # compute 
    index = np.r_[:nwind]
    #t = np.r_[:n]
    psd = np.zeros(nfft)
    SK_w = np.zeros([nfft2,K])
    for i in np.r_[:K]:
        xw = Window*x[index]
    	# filtering
        if filterr == 1:
            SK,_,_,_ = SK_W(xw,nfft2,Noverlap2,Window2)
            SK_w[:,i] = SK
            b = np.fft.fftshift(np.real(np.fft.ifft(SK)))
            xw = sg.fftconvolve(xw,b,mode='same') #xw = fftfilt(b,xw);
        xw = np.abs(sg.hilbert(xw)) # envelope
        xw = xw**2
        xw = xw - np.mean(xw)
        xw = sg.hilbert(xw)
        Xw = np.fft.fft(xw,nfft)/nfft
        psd = np.abs(Xw)**2 + psd
        index = index + (nwind - Noverlap)
    # normalize
    KMU = K*LA.norm(Window)**2;	#Normalizing scale factor ==> asymptotically unbiased
    psd = psd/KMU
    
    freq = np.fft.fftfreq(psd.shape[-1])
    psd = psd[:int(len(psd)/2)]
    freq = freq[:int(len(freq)/2)]
    return psd,freq,K,SK_w