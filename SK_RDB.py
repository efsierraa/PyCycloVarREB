# %% Loading
import os # change dir
Wdir  = os.getcwd()
Wdir2 = Wdir
import sys 
sys.path.append(Wdir+'\functions')# add to path
import numpy as np # np matrix functions
import matplotlib.pyplot as plt # plotting pack
from matplotlib.gridspec import GridSpec # easy subplots
import scipy.signal as sg # signal pack
from scipy.io import wavfile # read wav files
import scipy.io as sio # load save matlab files
from functions.func_sig_cost import tTacho_fsig,COT_intp,SK_W,PSD_envW,\
printxtips,printxtipsusr
# %% Loading DB
lfile     = 'G:\Mon Drive\GCPDS\Data Base\Surveillance8 Contest'
fs, data  = wavfile.read(lfile+'\Acc2.wav')
fs, tacho = wavfile.read(lfile+'\Tacho.wav')
#data  = data[:100*fs]
#tacho = tacho[:100*fs]

TPT = 44;isencod=1;
rpmt,tTacho,_ = tTacho_fsig(tacho,fs,TPT,isencod)
f0 = rpmt/60;del rpmt  # rpm to Hz

##% decimating signal
#data      = sg.decimate(data,2**2)
#f0        = sg.decimate(f0,2**2)
#fs        = fs/2**2
#f0        = f0[int(fs/2):-int(fs/2)]
#data      = data[int(fs/2):-int(fs/2)]
#_,tTacho,_= tTacho_fsig(f0,fs,TPT,0,1)
#%% angular resampling
SampPerRev = 2**np.fix(np.log2(fs/2/max(f0[int(fs/2):-int(fs/2)])))*2**2

datas,tc,SampPerRev = COT_intp(data,tTacho,TPT,fs,SampPerRev)

#% Save dataz
result_dict = {'datas': datas,'tc': tc,'SampPerRev':SampPerRev,
               'TPT':TPT,'fs':fs,'isencod':isencod}
sio.savemat('Dataz/D10_SK_angle_domain.mat', 
                result_dict,do_compression=True)
#%% loading signal SK - angle resampled - baseline 00
#% SK filtering Jerome
Nw       = 2**np.fix(np.log2(fs*1/.4))*2**3  # 1/2 minimun expected frq 2Hz
Nw       = int(Nw)
Nfft     = 2*Nw
Noverlap = round(3/4*Nw)

SK,_,_,_ = SK_W(data,Nfft,Noverlap,Nw)
b        = np.fft.fftshift(np.real(np.fft.ifft(SK)))
datas   = sg.fftconvolve(data,b,mode='same')

# % angular resampling 
isencod=1
SampPerRev = 2**np.fix(np.log2(fs/2/max(f0[int(fs/2):-int(fs/2)])))*2**2
datas,tc,SampPerRev = COT_intp(datas,tTacho,TPT,fs,SampPerRev)

# envelope spectrum
datae = sg.hilbert(datas)
datae = np.abs(datas)**2
datae = datae-np.mean(datae)
sp = np.abs(np.fft.fft(datae,Nfft)/Nfft)**2
freq = np.fft.fftfreq(sp.shape[-1])

sp = sp[:int(len(sp)/2)]
freq = freq[:int(len(freq)/2)]
freq = freq*SampPerRev

#% Save dataz
result_dict = {'sp': sp,'freq': freq,'datas':datas,'tc':tc,\
               'SampPerRev':SampPerRev}
sio.savemat('Dataz/D10_PSD_SK_baseline00.mat', 
                result_dict,do_compression=True)

# %% plot loading signal SK - angle resampled - baseline 00
lfile = ['D10_PSD_SK_baseline00']
mat_contents = sio.loadmat('Dataz/'+lfile[0], mdict=None, appendmat=True)

for k, v in mat_contents.items():
    try:
        if v.size==sum(v.shape)-1:
            globals()[k]=v.flatten()
        else:
            globals()[k]=v
    except:
        globals()[k]=v
del k,v,mat_contents,lfile

plt.rc('font', size=14) # default text size
fig1 = plt.figure()
plt.plot(freq, sp)
plt.axis([freq[0],100,0,1.1*max(sp[(freq>7)*(freq<100)])])
plt.xlabel('Order [xn]')
plt.ylabel('PSD')
plt.legend([r"$\left|X(\Theta)\right|^2$"])

printxtips(freq,sp,xheigh0=7.81055,deltapsd=1/2**2,K=12)
plt.show()
plt.savefig(Wdir2+'/Figures/exp05_envelopeSP_baseline00.pdf',bbox_inches='tight')
#%% loading signal angle resampled - SK baseline
lfile = ['D10_SK_angle_domain']
mat_contents = sio.loadmat('Dataz/'+lfile[0], mdict=None, appendmat=True)

for k, v in mat_contents.items():
    try:
        if v.size==sum(v.shape)-1:
            globals()[k]=v.flatten()
        else:
            globals()[k]=v
    except:
        globals()[k]=v
del k,v,mat_contents,lfile

#% SK filtering Jerome
Nw       = 2**np.fix(np.log2(1/.4*SampPerRev))*2**3
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
sp = np.abs(np.fft.fft(datae,Nfft)/Nfft)**2
freq = np.fft.fftfreq(sp.shape[-1])

sp = sp[:int(len(sp)/2)]
freq = freq[:int(len(freq)/2)]
freq = freq*SampPerRev

#% Save dataz
result_dict = {'sp': sp,'freq': freq}
sio.savemat('Dataz/D10_PSD_SK_baseline.mat', 
                result_dict,do_compression=True)
# %% plot loading signal angle resampled - SK baseline
lfile = ['D10_PSD_SK_baseline']
mat_contents = sio.loadmat('Dataz/'+lfile[0], mdict=None, appendmat=True)

for k, v in mat_contents.items():
    try:
        if v.size==sum(v.shape)-1:
            globals()[k]=v.flatten()
        else:
            globals()[k]=v
    except:
        globals()[k]=v
del k,v,mat_contents,lfile

fig1 = plt.figure()
plt.plot(freq, sp)
plt.axis([freq[0],100,0,1.1*max(sp[(freq>7)*(freq<100)])])
plt.xlabel('Order [xn]')
plt.ylabel('PSD')
plt.legend([r"$\left|X(\Theta)\right|^2$"])

printxtips(freq,sp,xheigh0=7.81055,deltapsd=1/2**2,K=12)
plt.show()
plt.savefig(Wdir2+'/Figures/exp05_envelopeSP_baseline.pdf',bbox_inches='tight')
#%% loading signal angle resampled - SK filtering proposal
lfile = ['D10_SK_angle_domain']
mat_contents = sio.loadmat('Dataz/'+lfile[0], mdict=None, appendmat=True)

for k, v in mat_contents.items():
    try:
        if v.size==sum(v.shape)-1:
            globals()[k]=v.flatten()
        else:
            globals()[k]=v
    except:
        globals()[k]=v
del k,v,mat_contents,lfile

#% SK filtering proposal
Nw1      = 2**np.fix(np.log2(1/.4*SampPerRev))*2**(5+3)
Nw1      = int(Nw1)
Nw2      = 2**np.fix(np.log2(1/.4*SampPerRev))*2**3
Nw2      = int(Nw2)
Nfft     = 2*Nw1
Nfft2    = 2*Nw2
# 7.667 evp failure found 7.759 evp theoretical
#Noverlap = round(.95*Nw1)#68-95-s997
Noverlap = round(3/4*Nw1)
#Noverlap = round(.25*Nw1)
#Noverlap = 0
Window   = np.kaiser(Nw1,beta=0) # beta 0 rectangular,5	Similar to a Hamming
# 6	Similar to a Hanning, 8.6	Similar to a Blackman

#Window   = np.ones(Nw1) # First window for SK per segments
filterr  = 1 # filtering with SK
psd,f,K  = PSD_envW(datas,Nfft,Noverlap,Window,Nw2,Nfft2,filterr)
f        = f*SampPerRev

#% Save dataz
result_dict = {'psd': psd,'f': f,'K':K}
sio.savemat('Dataz/D10_PSD_SK_proposal_75.mat', 
                result_dict,do_compression=True)

# %% plot signal angle resampled - SK filtering proposal
lfile = ['D10_PSD_SK_proposal_95']
mat_contents = sio.loadmat('Dataz/'+lfile[0], mdict=None, appendmat=True)

for k, v in mat_contents.items():
    try:
        if v.size==sum(v.shape)-1:
            globals()[k]=v.flatten()
        else:
            globals()[k]=v
    except:
        globals()[k]=v
del k,v,mat_contents,lfile

fxh =62/61
dfxh = 2e-3
px = max(psd[(f>fxh-dfxh)*(f<fxh+dfxh)])
fx = f[psd==px]
f = f/fx

fig2 = plt.figure()
plt.plot(f, psd)
plt.xlabel('Order [xn]')
plt.ylabel('PSD')
plt.axis([f[0],100,0,1.1*max(psd[(f>7)*(f<100)])])
plt.legend([r"$\left|X(\Theta)\right|^2$"])

xtip,_=printxtips(f,psd,xheigh0=7.68,deltapsd=1/2**5,K=12)
plt.show()
plt.savefig(Wdir2+'/Figures/exp05_envelopeSP_proposal.pdf',bbox_inches='tight')

# %% plot signal angle resampled - SK filtering proposal bands
lfile = ['D10_PSD_SK_proposal_95']
mat_contents = sio.loadmat('Dataz/'+lfile[0], mdict=None, appendmat=True)

for k, v in mat_contents.items():
    try:
        if v.size==sum(v.shape)-1:
            globals()[k]=v.flatten()
        else:
            globals()[k]=v
    except:
        globals()[k]=v
del k,v,mat_contents,lfile
fxh =62/61
dfxh = 2e-3
px = max(psd[(f>fxh-dfxh)*(f<fxh+dfxh)])
fx = f[psd==px]
f = f/fx

fig2 = plt.figure()
plt.plot(f, psd)
plt.xlabel('Order [xn]')
plt.ylabel('PSD')

fcage = 0.545
# finding max around 7.7
fxh  = 7.7
dfxh = 2e-1
px = max(psd[(f>fxh-dfxh)*(f<fxh+dfxh)])
fx = f[psd==px]

K = 5
fcage = 0.54527385
xtip = np.zeros(K)
xtip[0] = fx-fcage
for k in np.r_[:K]-2:
    xtip[k+2],_ =printxtipsusr(f,psd,xheigh=xtip[k+2],deltapsd=2e-2,tips='x',prt=1)
    try:
        xtip[k+3] = xtip[k+2]+fcage/2
    except IndexError:
        print('Are u finish?')
pxtip = int((len(xtip)-1)/2)
plt.axis([xtip[pxtip]-2,xtip[pxtip]+2,0,1.1*max(psd[(f>7)*(f<9)])])

plt.legend([r"$\left|X(\Theta)\right|^2$"])
plt.show()
plt.savefig(Wdir2+'/Figures/exp05_envelopeSP_proposal_zoom.pdf',bbox_inches='tight')