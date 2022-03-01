# Cyclostationary analysis in angular domain for bearing fault identification
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3102/)
[![Poetry](https://img.shields.io/badge/poetry-1.1.11-red.svg)](https://python-poetry.org/docs/master/#installation/)

The following repository has the functions needed in order to reproduce the results of the work entitled "Cyclo-non-stationary analysis for bearing fault identification based on instantaneous angular speed estimation" presented in the https://survishno.sciencesconf.org/, the database is available as supplementary material of the article, "Feedback on the Surveillance 8 challenge: Vibration-based diagnosis of a Safran aircraft engine", founded in https://doi.org/10.1016/j.ymssp.2017.01.037. If you find this material useful, please give a citation to the papers mentioned above.

# REB Highlight
Functions to REB failure identification based on Spectral kurtosis:

  SK_W # Spectral kurtosis from a Matlab repository made by ANTONI Jerome
  
  PSD_envW # Proposed algorithm sliding SK based implementation.
  
  bearingfault # Bearing fault numerical signal for any questions please contact http://lva.insa-lyon.fr/en/
## Example
``` python
mfreq = 0.4*fr # fr rotational speed
Nw1      = 2**np.ceil(np.log2(1/mfreq*fs))*2**3 # greater than Nw2 window for envelope spectrum
Nw1      = int(Nw1)
Nw2      = 2**8 # window for computation of SK
Nw2      = int(Nw2)
Nfft     = 2*Nw1
Nfft2    = 2*Nw2
Noverlap = round(3/4*Nw1)
Window   = np.kaiser(Nw1,beta=0) # beta 0 rectangular,5	Similar to a Hamming
# 6	Similar to a Hanning, 8.6	Similar to a Blackman

#%
filterr  = 1 # filtering with SK
psd,f,K,SK_w  = PSD_envW(x,Nfft,Noverlap,Window,Nw2,Nfft2,filterr)
```
  # Angular resampling functions
  These functions are translated/adapted from the book Noise and Vibration Analysis Signal Analysis and Experimental Procedures by Anders Brandt
  
  tTacho_fsig # Time indexes for angular resampling, and the Instantaneous Angular Profile (IAS), uses as input the tachometer signal
  
  COT_intp # interpolates a vibration signal with given time indexes in tTacho_fsig
  
  COT_intp2 # Interpolates the signal given an IAS profile, useful for numerical tests
  
  tTacho_fsigLVA # Creates an IF profile from the tachometer signal, translated from matlab from courses of the http://lva.insa-lyon.fr/en/
  
  # functions to print data-tips using matplotlib
  The following functions are using to put data-tips in a certain amount of harmonics given a Power Spectral Density
  
  printxtips
  
  printxtipsusr

  
  # References
  ```
  Sierra-Alonso, E. F., Caicedo-Acosta, J., Orozco Gutiérrez, Á. Á., Quintero, H. F., & Castellanos-Dominguez, G. (2021). Short-time/-angle spectral analysis for vibration monitoring of bearing failures under variable speed. Applied Sciences, 11(8), 3369.

  Sierra-Alonso, E. F., Antoni, J., & Castellanos-Dominguez, G. (2019, July). Cyclo-non-stationary analysis for bearing fault identification based on instantaneous angular speed estimation. In Surveillance, Vishno and AVE conferences.
```
## Revised version of the code
  2022-02-28, revised functions to enhance impulsive components using Spectral Kurtosis in the folder filter_SK, possible problem with function sg.fftconvolve inside PSD_envW, if you encounter a problem replace by sg.filtfilt(b, 1, x).
