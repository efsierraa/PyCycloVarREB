# Introduction

The following repository has the functions needed in order to reproduce the results of the work entitled "Filtered envelope spectrum using short periodograms for bearing fault identification under variable speed" presented in the http://iftomm2019.com/, the database is available as supplementary material of the article, "Feedback on the Surveillance 8 challenge: Vibration-based diagnosis of a Safran aircraft engine", founded in https://doi.org/10.1016/j.ymssp.2017.01.037. If you find this material useful, please give a citation to the papers mentioned above.

# REB_Highlight
Functions to REB failure identification based on Spectral kurtosis:

  SK_W # Spectral kurtosis from a Matlab repository made by ANTONI Jerome
  
  PSD_envW # Proposed algorithm SK based implementation.
  
  bearingfault # Bearing fault numerical signal for any questions please contact http://lva.insa-lyon.fr/en/
  
  # Angular resampling functions
  These functions are founded initially in Noise and Vibration Analysis Signal Analysis and Experimental Procedures by Anders Brandt
  
  tTacho_fsig # Time indexes for angular resampling, as the Instantaneous Angular Profile (IAS), uses as input the tachometer signal
  
  COT_intp # interpolates a vibration signal with given time indexes in tTacho_fsig
  
  COT_intp2 # Interpolates the signal given an IAS profile, useful for numerical tests.
  
  # functions to print data-tips using matplotlib
  The following functions are using to put data-tips in a certain amount of harmonics given a Power Spectral Density
  
  printxtips
  
  printxtipsusr
