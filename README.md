These codes perform a Bayesian Estimation of Multiple Species
(BEAMS) procedure (Kunz, Bassett & Hlozek 2007) on supernova data.
We simultaneously determine Type Ia supernova and core-collapse
supernova distances in a sample with a mixture of both.

Dependencies

numpy, scipy, emcee (http://dan.iel.fm/emcee/current/user/install/)

There's no real install file yet, just put the BEAMS directory in your PATH.

snbeams.py -h

will provide the rest of the documentation.

------------
# EXAMPLE A:

  Run BEAMS on the example data using only Type Ia Supernovae and
  compare to NN classification.  Use just a few steps for this example,
  but for robust results the nsteps keyword should be at least 1,000.
  Results are also most robust with the parallel-tempered ensemble sampler
  (ntemps = 20 works well), but that takes longer to run.

   ./snbeams.py -p BEAMS.params -i exampledata/PS1_SIM_example.FITRES -o exampledata/PS1_BEAMS.SNIa.out --onlyIa --piacol PTRUE_Ia --ninit 50 --nsteps 200
   ./snbeams.py -p BEAMS.params -i exampledata/PS1_SIM_example.FITRES -o exampledata/PS1_BEAMS.NN.out --piacol PNN_Ia --ninit 50 --nsteps 200

  BEAMS then produces 3 files, using the fileroot of the output file
  you specify: the basic output file, a file in FITRES format, where the
  SN Ia distances have been converted to mB/mBERR values
  (distance mod. - 19.36), and a distance modulus covariance matrix.

  in python:

     import pylab as plt
     from astropy.cosmology import Planck13 as cosmo
     from BEAMS.txtobj import txtobj
     true = txtobj('exampledata/PS1_BEAMS.SNIa.out')
     nn = txtobj('exampledata/PS1_BEAMS.NN.out')

     plt.errorbar(true.zCMB,true.popAmean-cosmo.distmod(true.zCMB).value,
                  yerr=true.popAmean_err,fmt='o')
     plt.errorbar(nn.zCMB,true.popAmean-cosmo.distmod(nn.zCMB).value,
                  yerr=nn.popAmean_err,fmt='o')
     plt.ylim([-0.1,0.1])
     plt.xlim([0.01,0.7])
     plt.xscale('log')
     plt.xlabel('$z$')
     plt.ylabel('Hubble Residual')

  the results look great!  Unfortunately, they don't mean anything yet.
  You really need to use the parallel-tempered ensemble sampler - or 
  turn of the scale and shift parameters in mcmc.params - and use many 
  more steps if you want to trust the results.
