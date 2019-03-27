# REB_Highlight
Functions to REB failure identification based on Spectral kurtosis:

  SK_W # Spectral kurtosis from a Matlab reposository made by ANTONI Jerome
  
  PSD_envW # Proposed algorithm SK based implementation
  
  bearingfault # Bearing fault numerical signal for any questions pleas contact http://lva.insa-lyon.fr/en/
  
  # Angular resampling functions
  This functions are originally founded in Noise and Vibration Analysis Signal Analysis and Experimental Procedures by Anders Brandt
  
  tTacho_fsig # Time indexes for angular resmapling, as the Instantaneus Angular Profile (IAS), uses as input the tachometer signal
  
  COT_intp # interpolates a vibration signal with a given time indexes in tTacho_fsig
  
  COT_intp2 # Interpolates the signal given an IAS profile, usefull for numerical tests
  
  # functions to print data-tips using matplotlib
  The following functions are using to put data-tips in a certain amount of harmonics given a Power Spectral Density
  
  printxtips
  
  printxtipsusr
