# signal_kharmdb_pink.m
import numpy as np # math pack!
import scipy.signal as sg # DSP
from scipy import integrate # integration pack
from numpy import linalg as LA # Linear algebra
from scipy import interpolate # interpolate lib
import sys # sio pack
import matplotlib.pyplot as plt # plotting 
import time # tic toc
import scipy.optimize as optm # optimization pack
import scipy.io as sio # load save matlab files

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
# % PSD filtered welch Spectral Kurtosis
def PSD_envW(x,nfft,Noverlap,Window,Nwind2,nfft2,filterr):

    Window = Window.reshape(-1,1)[:,0]
    n = len(x)		# Number of data points
    nwind = len(Window) # length of window
    x = x.reshape(-1,1)[:,0]
    K = np.fix((n-Noverlap)/(nwind-Noverlap)) # Number of windows
    Noverlap2 = int(np.round(3/4*Nwind2))
    Window2 = np.hanning(Nwind2)
    
    # compute 
    index = np.r_[:nwind]
    #t = np.r_[:n]
    psd = np.zeros(nfft)
    for i in np.r_[:K]:
        xw = Window*x[index]
    	# filtering
        if filterr == 1:
            SK,_,_,_ = SK_W(xw,nfft2,Noverlap2,Window2)
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
    return psd,freq,K
#%% Bearing fault model
def bearingfault(dsurD, N, fo,fi,typeF,fr,fe,T):
#[out,TPT] = bearingfault(dsurD, N, fo,fi,type,fr,fe,T)
#dsurD : rapport entre diametre de bille et primitif
#N : nombre de billes
#fo : fréquence de rotation bague externe (si vecteur 2*1, fréquence début et fin)
#fi : fréquence de rotation bague interne (si vecteur 2*1, fréquence début et fin)
#type : 0:'BPFO' pour défaut de bague externe
#       1:'BPFI' pour défaut de bague interne
#fr : fréquence de résonance structure
#fe : fréquence d'échantillonnage
#T : durée du signal généré
    TPT=44;
    eps=0.2;
    t = np.linspace(0,T,T*fe)

    if not isinstance(fo,int):
        Fo = fo[0] + t*(np.diff(fo)/T)
    else:
        Fo=np.ones(t.shape)*fo
        
    if not isinstance(fi,int):
        Fi = fi[0] + t*(np.diff(fi)/T)
    else:
        Fi=np.ones(t.shape)*fi
        
    Fc= 1/2*((1-dsurD)*Fi + (1+dsurD)*Fo)
    
    tetao =np.cumsum(Fo)/fe
    tetai =np.cumsum(Fi)/fe
    tetac =np.cumsum(Fc)/fe
    
    out = np.zeros([2*len(t),2])
    out[:len(t),0] = np.sin(2*np.pi*TPT*(tetao-tetai))
    
    timpend = np.log(100)/(eps*2*np.pi*fr)
    timp = np.linspace(0,timpend,timpend*fe)
    
    impulse = np.sin(2*np.pi*fr*timp)*np.exp(-eps*2*np.pi*fr*timp)
    #impulse = zeros(size(timp));
    #impulse(1)=1;
    
    nimpulse =len(impulse)
    BPFO=N/2*(1-dsurD)
    BPFI=N/2*(1+dsurD)
    
    print('BPFO=',np.abs(BPFO))
    print('BPFI=',np.abs(BPFI))
    
    idef,modulation = faill_type(typeF,N,tetac,tetao,tetai)
    #idef = round(tdef*fe)+1;
    
    for ii in np.r_[:len(idef)]:
        out[idef[ii]+ np.r_[:nimpulse],1]=impulse.reshape(-1,1)[:,0] #+ randn(nimpulse,1)
        
    out = out[:len(t),:]
    
    out[:,1] = out[:,1]*modulation
    
    # bruit
    #out(:,2) = out(:,2) + randn(size(out(:,2))).*std(out(:,2));
    return out,TPT
#% auxiliar functions switch bearing failure model
def BPFO(N,tetac,tetao,tetai):
    idef=np.flatnonzero(np.abs(np.diff(np.round(N*(tetac-tetao)))))
    modulation = np.cos(2*np.pi*tetao.reshape(-1,1)[:,0])
    return idef,modulation

def BPFI(N,tetac,tetao,tetai):
    idef=np.flatnonzero(np.abs(np.diff(np.round(N*(tetac-tetai)))))
    modulation = np.cos(2*np.pi*tetai.reshape(-1,1)[:,0])
    return idef,modulation

def faill_type(argument,N,tetac,tetao,tetai):
    switcher = {
    0: BPFO,
    1: BPFI
    }
    # Get the function from switcher dictionary
    func = switcher.get(argument,"Press 0 or 1")
    # Execute the function
    return func(N,tetac,tetao,tetai)
#%% Tacho grid 2 interpolate
def tTacho_fsig(rpm,fs,PPR,isencod):
    # rpm tacho signal
    # fs sampling frequency
    # PPR pulses per revolition, resolution tacho 44 surveillance case
    # isencod ==1 encoder signal or IF profile
    # sm smoothing intervals with 20
    if isencod==1:
        x=rpm
    else: 
        x=np.sin(2*np.pi*PPR*integrate.cumtrapz(rpm)/fs)
    t=np.arange(0,len(x))
    t=t/fs
    # Produce +1 where signal is above trigger level
    # and -1 where signal is below trigger level
    TLevel=0
    xs=np.sign(x-TLevel)
    # Differentiate this to find where xs changes
    # between -1 and +1 and vice versa
    xDiff=np.diff(xs);
    # We need to synchronize xDiff with variable t from the
    # code above, since DIFF shifts one step
    tDiff=t[1:]
    # Now find the time instances of positive slope positions
    # (-2 if negative slope is used)
    tTacho=tDiff[xDiff.T == 2] # xDiff.T return the indexes boolean
    # Count the time between the tacho signals and compute
    # the RPM at these instances
    rpmt=60/PPR/np.diff(tTacho); # Temporary rpm values
    # Use three tacho pulses at the time and assign mean
    # value to the center tacho pulse
    rpmt=0.5*(rpmt[0:-1]+rpmt[1:]);
    tTacho=tTacho[1:-1]; # diff again shifts one sample
    
    wfiltsv = int(2**np.fix(np.log2(.05*fs)))-1
    rpmt= sg.savgol_filter(rpmt,wfiltsv,2)# smoothing filter
    #rpmt= sg.medfilt(rpmt,2**3-1)
    
    rpmt=interpolate.InterpolatedUnivariateSpline\
    (tTacho,rpmt,w=None, bbox=[None, None], k=1)
    
    rpmt = rpmt(t)
    return rpmt,tTacho,xs
#% Angular resampling given tTacho
def COT_intp(x,tTacho,PPR,fs,SampPerRev):
    tTacho=tTacho[::PPR] # Pick out every PPR pulse
    ts=np.r_[:0] # Synchronous time instances
    
    for n in np.r_[:len(tTacho)-1]:
        tt=np.linspace(tTacho[n],tTacho[n+1],SampPerRev+1)
        ts=np.r_[ts,tt[:-1]]
    # Now upsample the original signal 10 times (to a total
    # of approx 25 times oversampling).
    x=sg.resample(x,10*len(x))
    fs=10*fs
    #create a time axis for this upsampled signal
    tx=np.r_[:len(x)]/fs
    # Interpolate x onto the x-axis in ts instead of tx
    xs=interpolate.InterpolatedUnivariateSpline(tx,x,w=None, bbox=[None, None], k=1)
    xs=xs(ts)
    tc=np.r_[:len(xs)/SampPerRev:1/SampPerRev]
    return xs,tc,SampPerRev

#% COT second function for IF as input
def COT_intp2(f_ins,SampPerRev,x,fs):
    t= np.r_[::len(x)];t=t/fs;
    # Calculate the inst. angle as function of time
    # (in part of revolutions, not radians!)
    Ainst = integrate.cumtrapz(f_ins,t)
    # Find every 1/SampPerRev of a cycle in Ainst
    minA = min(Ainst)
    maxA = max(Ainst)
    Fractions = np.r_[np.ceil(minA*SampPerRev)/SampPerRev:maxA:1/SampPerRev]
    # New sampling times
    tt=interpolate.InterpolatedUnivariateSpline(Ainst,t,\
                                                w=None, bbox=[None, None], k=1)
    tt = tt(Fractions)
    # Now upsample the original signal 10 times (to a total
    # of approx 25 times oversampling)
    x = sg.resample(x,10)
    fs= 10*fs
    # create a time axis for this upsampled signal
    #tx=(0:1/fs:(length(x)-1)/fs);
    tx= np.r_[:len(x)]/fs
    # Interpolate x onto the x-axis in ts instead of tx
    xs=interpolate.InterpolatedUnivariateSpline(tx,x,\
                                                w=None, bbox=[None, None], k=1)
    xs = xs(tt)
    tc=np.r_[:len(xs)]/SampPerRev
    return xs,tc,SampPerRev,tt
# encoder signal to IAS
def tTacho_fsigLVA(top,fs,TPT=44):
    top = top/max(abs(top))
    
    t = np.r_[:len(top)]/fs
    seuil=0;
    ifm=np.where((top[:-1]<=seuil) & (top[1:]>seuil))[0]
    tfm = (t[ifm]*top[ifm+1]-t[ifm+1]*top[ifm])/(top[ifm+1] -top[ifm])
    ifm = tfm*fs;
    
    W =np.array([])
    tW=np.array([])
    for ii in np.r_[:TPT]:
        tfmii = tfm[ii::TPT];
        W=np.append(W,[1/np.diff(tfmii)])
        tW=np.append(tW,[tfmii[:-1]/2 + tfmii[1:]/2])
    
    ordre=np.argsort(tW)
    tW=np.sort(tW)
    W = W[ordre]
    
    W=interpolate.InterpolatedUnivariateSpline\
    (tW,W,w=None, bbox=[None, None])
    
    W = W(t)
    return t,W
#%% plotting functions
def printxtips(f,psd,xheigh0,deltapsd,K):
    txheigh = np.zeros(K)
    xheigh  = np.zeros(K)
    for k in np.r_[:K]:    
        tips = 'x'+str(k+1)
        xheigh[k] = xheigh0*(k+1)
        txheigh[k] = max(psd[(f>xheigh[k]-deltapsd)*(f<xheigh[k]+deltapsd)])
        if k==0:
            xheigh0 = f[psd==txheigh[k]]
            xheigh[k] = xheigh0*(k+1)
        plt.text(xheigh[k],txheigh[k],tips)
    return xheigh,txheigh
def printxtipsusr(f,psd,xheigh,deltapsd,tips,prt):
    # f,psd spectrum
    # xheigh, deltapsd x-coordinate for look max, deltapsd interval to search
    # tips marker
    # prt if prt==1 print xheigh ie x position
    txheigh = max(psd[(f>xheigh-deltapsd)*(f<xheigh+deltapsd)])
    xheigh = f[psd==txheigh]
    if prt == 1:
        tips = str(np.round(xheigh,2)[0])
        plt.text(xheigh,txheigh,tips)
    else:
        plt.text(xheigh,txheigh,tips)
    return xheigh,txheigh
