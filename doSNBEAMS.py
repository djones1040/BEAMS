#!/usr/bin/env python
# D. Jones - 9/1/15
"""BEAMS method for PS1 data"""
import numpy as np
from scipy.misc import logsumexp
from scipy.special import erf
from astropy.cosmology import Planck13 as cosmo
from scipy.stats import norm

class BEAMS:
    def __init__(self):
        self.clobber = False
        self.verbose = False

    def add_options(self, parser=None, usage=None, config=None):
        import optparse
        if parser == None:
            parser = optparse.OptionParser(usage=usage, conflict_handler="resolve")

        # The basics
        parser.add_option('-v', '--verbose', action="count", dest="verbose",default=1)
        parser.add_option('--debug', default=False, action="store_true",
                          help='debug mode: more output and debug files')
        parser.add_option('--clobber', default=False, action="store_true",
                          help='clobber output file')
        parser.add_option('--append', default=False, action="store_true",
                          help='open output file in append mode')

        if config:
            # Input file
            parser.add_option('--pacol', default=config.get('doSNBEAMS','pacol'), type="string",
                              help='column in input file used as prior P(A)')
            parser.add_option('--mucol', default=config.get('doSNBEAMS','mucol'), type="string",
                              help='column name in input file header for distance modulus')
            parser.add_option('--muerrcol', default=config.get('doSNBEAMS','muerrcol'), type="string",
                              help='column name in input file header for distance modulus errors')
            parser.add_option('--zcol', default=config.get('doSNBEAMS','zcol'), type="string",
                              help='column name in input file header for z')


            # population A guesses and priors (the pop we care about)
            parser.add_option('--popAguess', default=map(float,config.get('doSNBEAMS','popAguess').split(',')),type='float',
                              help='comma-separated initial guesses for population A: mean, standard deviation',nargs=2)
            parser.add_option('--popAprior_mean', default=map(float,config.get('doSNBEAMS','popAprior_mean').split(',')),type='float',
                              help="""comma-separated gaussian prior for mean of population A: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--popAprior_std', default=map(float,config.get('doSNBEAMS','popAprior_std').split(',')),type='float',
                              help="""comma-separated gaussian prior for std dev. of population A: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--popAfixed', default=map(float,config.get('doSNBEAMS','popAfixed').split(',')),type='float',
                              help="""comma-separated values for population A params: mean, standard deviation.  
For each param, set to 0 to include in parameter estimation, set to 1 to keep fixed""",nargs=2)


            # population B guesses and priors (the pop we care about)
            parser.add_option('--popBguess', default=map(float,config.get('doSNBEAMS','popBguess').split(',')),type='float',
                              help='comma-separated initial guesses for population B: mean, standard deviation',nargs=2)
            parser.add_option('--popBprior_mean', default=map(float,config.get('doSNBEAMS','popBprior_mean').split(',')),type='float',
                              help="""comma-separated gaussian prior for mean of population B: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--popBprior_std', default=map(float,config.get('doSNBEAMS','popBprior_std').split(',')),type='float',
                              help="""comma-separated gaussian prior for std dev. of population B: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--popBfixed', default=map(float,config.get('doSNBEAMS','popBfixed').split(',')),type='float',
                              help="""comma-separated values for population B params: mean, standard deviation.  
For each param, set to 0 to include in parameter estimation, set to 1 to keep fixed""",nargs=2)

            # population B guesses and priors for second gaussian (the pop we care about)
            parser.add_option('--popB2guess', default=map(float,config.get('doSNBEAMS','popB2guess').split(',')),type='float',
                              help='comma-separated initial guesses for population B: mean, standard deviation',nargs=2)
            parser.add_option('--popB2prior_mean', default=map(float,config.get('doSNBEAMS','popB2prior_mean').split(',')),type='float',
                              help="""comma-separated gaussian prior for mean of population B: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--popB2prior_std', default=map(float,config.get('doSNBEAMS','popB2prior_std').split(',')),type='float',
                              help="""comma-separated gaussian prior for std dev. of population B: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--popB2fixed', default=map(float,config.get('doSNBEAMS','popB2fixed').split(',')),type='float',
                              help="""comma-separated values for population B params: mean, standard deviation.  
For each param, set to 0 to include in parameter estimation, set to 1 to keep fixed""",nargs=2)

            # the skewness of the pop. B gaussian
            parser.add_option('--skewBguess', default=config.get('doSNBEAMS','skewBguess'),type='float',
                              help='comma-separated initial guesses for population B: mean, standard deviation')
            parser.add_option('--skewBprior', default=map(float,config.get('doSNBEAMS','skewBprior').split(',')),type='float',
                              help="""comma-separated gaussian prior for skewness of population B: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--skewBfixed', default=config.get('doSNBEAMS','skewBfixed'),type='int',
                              help="""set to 0 to include skewness in parameter estimation, set to 1 to keep fixed""")

            # fraction of contaminants: guesses and priors
            parser.add_option('--fracBguess', default=config.get('doSNBEAMS','fracBguess'),type='float',
                              help='initial guess for fraction of contaminants, set to negative to omit')
            parser.add_option('--fracBprior', default=map(float,config.get('doSNBEAMS','fracBprior').split(',')),type='float',
                              help="""comma-separated gaussian prior on fraction of contaminants: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--fracBfixed', default=config.get('doSNBEAMS','fracBfixed'),type='int',
                              help="""0 to return posterior frac. of contaminants, 1 to keep fixed""")

           # fraction of contaminants: guesses and priors for gaussian 2
            parser.add_option('--fracB2guess', default=config.get('doSNBEAMS','fracBguess'),type='float',
                              help='initial guess for fraction of contaminants, set to negative to omit')
            parser.add_option('--fracB2prior', default=map(float,config.get('doSNBEAMS','fracBprior').split(',')),type='float',
                              help="""comma-separated gaussian prior on fraction of contaminants: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--fracB2fixed', default=config.get('doSNBEAMS','fracBfixed'),type='int',
                              help="""0 to return posterior frac. of contaminants, 1 to keep fixed""")

            # options to fit to a second-order "step" effect on luminosity.
            # This is the host mass bias for SNe.
            parser.add_option('--plcol', default=config.get('doSNBEAMS','plcol'), type="string",
                              help='column in input file used as probability for an additional luminosity correction P(L)')
            parser.add_option('--lstep', default=config.get('doSNBEAMS','lstep'), type="int",
                              help='make step correction if set to 1 (default=%default)')
            parser.add_option('--lstepguess', default=config.get('doSNBEAMS','lstepguess'),type='float',
                              help='initial guesses for luminosity correction')
            parser.add_option('--lstepprior', default=map(float,config.get('doSNBEAMS','lstepprior').split(',')),type='float',
                              help="""comma-separated gaussian prior for mean of luminosity correction: centroid, sigma.  
For flat prior, use empty strings""",nargs=2)
            parser.add_option('--lstepfixed', default=config.get('doSNBEAMS','lstepfixed'),type='float',
                              help="""comma-separated values for luminosity correction.
For each param, set to 0 to include in parameter estimation, set to 1 to keep fixed""")

            # SALT2 SN parameters: guesses and priors
            parser.add_option('--salt2alphaguess', default=config.get('doSNBEAMS','salt2alphaguess'),type='float',
                              help='initial guess for fraction of contaminants, set to negative to omit')
            parser.add_option('--salt2alphaprior',
                              default=map(float,config.get('doSNBEAMS','salt2alphaprior').split(',')),type='float',
                              help="""comma-separated gaussian prior on fraction of contaminants: centroid, sigma.
For flat prior, use empty string""",nargs=2)
            parser.add_option('--salt2alphafixed', default=config.get('doSNBEAMS','salt2alphafixed'),type='int',
                              help="""0 to return posterior frac. of contaminants, 1 to keep fixed""")
            parser.add_option('--salt2betaguess', default=config.get('doSNBEAMS','salt2betaguess'),type='float',
                              help='initial guess for fraction of contaminants, set to negative to omit')
            parser.add_option('--salt2betaprior',
                              default=map(float,config.get('doSNBEAMS','salt2betaprior').split(',')),type='float',
                              help="""comma-separated gaussian prior on fraction of contaminants: centroid, sigma.
For flat prior, use empty string""",nargs=2)
            parser.add_option('--salt2betafixed', default=config.get('doSNBEAMS','salt2betafixed'),type='int',
                              help="""0 to return posterior frac. of contaminants, 1 to keep fixed""")

            parser.add_option('--salt2alpha', default=config.get('doSNBEAMS','salt2alpha'), type="float",
                              help='SALT2 alpha parameter from a spectroscopic sample (default=%default)')
            parser.add_option('--salt2alphaerr', default=config.get('doSNBEAMS','salt2alphaerr'), type="float",
                              help='nominal SALT2 alpha uncertainty from a spectroscopic sample (default=%default)')
            parser.add_option('--salt2beta', default=config.get('doSNBEAMS','salt2beta'), type="float",
                              help='nominal SALT2 beta parameter from a spec. sample (default=%default)')
            parser.add_option('--salt2betaerr', default=config.get('doSNBEAMS','salt2betaerr'), type="float",
                              help='nominal SALT2 beta uncertainty from a spec. sample (default=%default)')        

            # output and number of threads
            parser.add_option('--nthreads', default=config.get('doSNBEAMS','nthreads'), type="int",
                              help='Number of threads for MCMC')
            parser.add_option('--nwalkers', default=config.get('doSNBEAMS','nwalkers'), type="int",
                              help='Number of walkers for MCMC')
            parser.add_option('--nsteps', default=config.get('doSNBEAMS','nsteps'), type="int",
                              help='Number of steps for MCMC')
            parser.add_option('--ninit', default=config.get('doSNBEAMS','ninit'), type="int",
                              help="Number of steps before the samples wander away from the initial values and are 'burnt in'")


            parser.add_option('--nzbins', default=config.get('doSNBEAMS','nzbins'), type="int",
                              help='Number of z bins')
            parser.add_option('--zmin', default=config.get('doSNBEAMS','zmin'), type="float",
                              help='min redshift')
            parser.add_option('--zmax', default=config.get('doSNBEAMS','zmax'), type="float",
                              help='max redshift')

            # alternate functional models
            parser.add_option('--twogauss', default=map(int,config.get('doSNBEAMS','twogauss'))[0], action="store_true",
                              help='two gaussians for pop. B')
            parser.add_option('--skewedgauss', default=map(int,config.get('doSNBEAMS','skewedgauss'))[0], action="store_true",
                              help='skewed gaussian for pop. B')
            parser.add_option('--simcc', default=config.get('doSNBEAMS','simcc'), type="string",
                              help='if filename is given, construct a polynomial-altered empirical CC SN function')
            parser.add_option('--snpars', default=map(int,config.get('doSNBEAMS','snpars'))[0], action="store_true",
                              help='use BEAMS to constrain SALT2 alpha and beta')

            parser.add_option('-i','--inputfile', default=config.get('doSNBEAMS','inputfile'), type="string",
                              help='file with the input data')
            parser.add_option('-o','--outputfile', default=config.get('doSNBEAMS','outputfile'), type="string",
                              help='Output file with the derived parameters for each redshift bin')

        else:
            # Input file
            parser.add_option('--pacol', default='PA', type="string",
                              help='column in input file used as prior P(A)')
            parser.add_option('--mucol', default='mu', type="string",
                              help='column name in input file header for residuals')
            parser.add_option('--muerrcol', default='mu_err', type="string",
                              help='column name in input file header for residual errors')
            parser.add_option('--zcol', default='z', type="string",
                              help='column name in input file header for z')

            # population A guesses and priors (the pop we care about)
            parser.add_option('--popAguess', default=(0.0,0.1),type='float',
                              help='comma-separated initial guesses for population A: mean, standard deviation',nargs=2)
            parser.add_option('--popAprior_mean', default=(0.0,0.5),type='float',
                              help="""comma-separated gaussian prior for mean of population A: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--popAprior_std', default=(0.1,0.2),type='float',
                              help="""comma-separated gaussian prior for std dev. of population A: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--popAfixed', default=(0,0),type='float',
                              help="""comma-separated values for population A params: mean, standard deviation.  
For each param, set to 0 to include in parameter estimation, set to 1 to keep fixed""",nargs=2)


            # population B guesses and priors (the pop we care about)
            parser.add_option('--popBguess', default=(1.0,1.0),type='float',
                              help='comma-separated initial guesses for population B: mean, standard deviation',nargs=2)
            parser.add_option('--popBprior_mean', default=(1.0,1.0),type='float',
                              help="""comma-separated gaussian prior for mean of population B: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--popBprior_std', default=(1.0,0.5),type='float',
                              help="""comma-separated gaussian prior for std dev. of population B: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--popBfixed', default=(0,0),type='float',
                              help="""comma-separated values for population B params: mean, standard deviation.  
For each param, set to 0 to include in parameter estimation, set to 1 to keep fixed""",nargs=2)


            # population B guesses and priors for second gaussian (the pop we care about)
            # if we use a skewed gaussian, the pop B refers to the right side of a single gaussian
            parser.add_option('--popB2guess', default=(0.0,4.0),type='float',
                              help='comma-separated initial guesses for population B: mean, standard deviation',nargs=2)
            parser.add_option('--popB2prior_mean', default=(0.0,1.0),type='float',
                              help="""comma-separated gaussian prior for mean of population B: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--popB2prior_std', default=(4.0,5.0),type='float',
                              help="""comma-separated gaussian prior for std dev. of population B: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--popB2fixed', default=(0,0),type='float',
                              help="""comma-separated values for population B params: mean, standard deviation.  
For each param, set to 0 to include in parameter estimation, set to 1 to keep fixed""",nargs=2)

            # the skewness of the pop. B gaussian
            # the skewness of the pop. B gaussian
            parser.add_option('--skewBguess', default=0,type='float',
                              help='comma-separated initial guesses for population B: mean, standard deviation',nargs=2)
            parser.add_option('--skewBprior', default=(0,4),type='float',
                              help="""comma-separated gaussian prior for skewness of population B: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--skewBfixed', default=0,type='int',
                              help="""set to 0 to include skewness in parameter estimation, set to 1 to keep fixed""",nargs=2)


            # fraction of contaminants: guesses and priors
            parser.add_option('--fracBguess', default=0.05,type='float',
                              help='initial guess for fraction of contaminants, set to negative to omit')
            parser.add_option('--fracBprior', default=(0.0,0.1),type='float',
                              help="""comma-separated gaussian prior on fraction of contaminants: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--fracBfixed', default=0,type='int',
                              help="""0 to return posterior frac. of contaminants, 1 to keep fixed""")

            # fraction of contaminants: guesses and priors for gaussian 2
            parser.add_option('--fracB2guess', default=0.05,type='float',
                              help='initial guess for fraction of contaminants, set to negative to omit')
            parser.add_option('--fracB2prior', default=(0.05,0.1),type='float',
                              help="""comma-separated gaussian prior on fraction of contaminants: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--fracB2fixed', default=0,type='int',
                              help="""0 to return posterior frac. of contaminants, 1 to keep fixed""")


            # options to fit to a second-order "step" effect on luminosity.
            # This is the host mass bias for SNe.
            parser.add_option('--plcol', default='PL', type="string",
                              help='column in input file used as probability for an additional luminosity correction P(L) (default=%default)')
            parser.add_option('--lstep', default=0, type="int",
                              help='make step correction if set to 1 (default=%default)')
            parser.add_option('--lstepguess', default=0.07,type='float',
                              help='initial guesses for luminosity correction')
            parser.add_option('--lstepprior', default=(0.07,0.023),type='float',
                              help="""comma-separated gaussian prior for mean of luminosity correction: centroid, sigma.  
For flat prior, use empty strings""",nargs=2)
            parser.add_option('--lstepfixed', default=0,type='float',
                              help="""comma-separated values for luminosity correction.
For each param, set to 0 to include in parameter estimation, set to 1 to keep fixed""")

            # SALT2 SN parameters: guesses and priors
            parser.add_option('--salt2alphaguess', default=0.147,type='float',
                              help='initial guess for fraction of contaminants, set to negative to omit')
            parser.add_option('--salt2alphaprior', default=(0.147,0.1),type='float',
                              help="""comma-separated gaussian prior on fraction of contaminants: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--salt2alphafixed', default=0,type='int',
                              help="""0 to return posterior frac. of contaminants, 1 to keep fixed""")
            parser.add_option('--salt2betaguess', default=3.13,type='float',
                              help='initial guess for fraction of contaminants, set to negative to omit')
            parser.add_option('--salt2betaprior', default=(3.13,0.5),type='float',
                              help="""comma-separated gaussian prior on fraction of contaminants: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--salt2betafixed', default=0,type='int',
                              help="""0 to return posterior frac. of contaminants, 1 to keep fixed""")
        
            parser.add_option('--salt2alpha', default=0.147, type="float",
                              help='SALT2 alpha parameter from a spectroscopic sample (default=%default)')
            parser.add_option('--salt2alphaerr', default=0.01, type="float",
                              help='nominal SALT2 alpha uncertainty from a spectroscopic sample (default=%default)')
            parser.add_option('--salt2beta', default=3.13, type="float",
                              help='nominal SALT2 beta parameter from a spec. sample (default=%default)')
            parser.add_option('--salt2betaerr', default=0.12, type="float",
                              help='nominal SALT2 beta uncertainty from a spec. sample (default=%default)')
            
        
            # output and number of threads
            parser.add_option('--nthreads', default=20, type="int",
                              help='Number of threads for MCMC')
            parser.add_option('--nwalkers', default=200, type="int",
                              help='Number of walkers for MCMC')
            parser.add_option('--nsteps', default=1000, type="int",
                              help='Number of steps for MCMC')
            parser.add_option('--ninit', default=50, type="int",
                              help="Number of steps before the samples wander away from the initial values and are 'burnt in'")


            parser.add_option('--nzbins', default=30, type="int",
                              help='Number of z bins')
            parser.add_option('--zmin', default=0.02, type="float",
                              help='min redshift')
            parser.add_option('--zmax', default=0.7, type="float",
                              help='max redshift')

            # alternate functional models
            parser.add_option('--twogauss', default=False, action="store_true",
                              help='two gaussians for pop. B')
            parser.add_option('--skewedgauss', default=False, action="store_true",
                              help='skewed gaussian for pop. B')
            parser.add_option('--simcc', default=None, type='string',
                              help='if filename is given, construct a polynomial-altered empirical CC SN function')
            parser.add_option('--snpars', default=False, action="store_true",
                              help='use BEAMS to constrain SALT2 alpha and beta')


            parser.add_option('-i','--inputfile', default='BEAMS.input', type="string",
                              help='file with the input data')
            parser.add_option('-o','--outputfile', default='beamsCosmo.out', type="string",
                              help='Output file with the derived parameters for each redshift bin')

        parser.add_option('-p','--paramfile', default='', type="string",
                          help='fitres file with the SN Ia data')

        return(parser)

    def main(self,inputfile):
        from txtobj import txtobj
        import os

        inp = txtobj(inputfile)
        inp.PA = inp.__dict__[self.options.pacol]
        if self.options.simcc:
            simcc = txtobj(simcc)
        if self.options.lstep:
            inp.PL = inp.__dict__[self.options.plcol]
        else:
            self.options.lstepfixed = 1
            self.options.lstepguess = 0.0

        if not len(inp.PA):
            import exceptions
            raise exceptions.RuntimeError('Warning : no data in input file!!')            

        # open the output file
        if not os.path.exists(self.options.outputfile) or self.options.clobber:
            writeout = True
            fout = open(self.options.outputfile,'w')
            outline = "# muA muAerr muAerr_m muAerr_p sigA sigAerr sigAerr_m sigAerr_p muB muBerr muBerr_m muBerr_p "
            outline += "sigB sigBerr sigBerr_m sigBerr_p muB2 muB2err muB2err_m muB2err_p sigB2 sigB2err sigB2err_m "
            outline += "sigB2err_p skewB skewBerr skewBerr_m skewBerr_p fracB fracBerr fracBerr_m fracBerr_p fracB2 "
            outline += "fracB2err fracB2err_m fracB2err_p lstep lsteperr lsteperr_m lsteperr_p alpha alphaerr alphaerr_m "
            outline += "alphaerr_p beta betaerr betaerr_m betaerr_p"
            print >> fout, outline
            fout.close()
        elif not self.options.append:
            writeout = False
            print('Warning : files %s exists!!  Not clobbering'%self.options.outputfile)

        # run the MCMC
        zcontrol = np.logspace(np.log10(self.options.zmin),np.log10(self.options.zmax),num=self.options.nzbins)
        cov,params,samples = self.mcmc(inp,zcontrol)

        root,ext = os.path.splitext(self.options.outputfile)
        fout = open('%s.covmat'%root,'w')
        print >> fout, '%i'%len(cov)
        shape = np.shape(cov)[0]
        for i in range(shape):
            outline = ''
            for j in range(shape):
                outline += '%8.5e '%cov[j,i]
                if i != j:
                    print >> fout, '%8.5e'%cov[j,i]#outline
                else:
                    print >> fout, '%8.5e'%0 #outline
        fout.close()

        # residA,sigA,residB,sigB,fracB,lstep = self.mcmc(inp)
        sigA,residB,sigB,residB2,sigB2,skewB,fracB,fracB2,lstep,alpha,beta = params[:11]
        mupar = params[11:]

        chain_len = len(samples[:,0])

        count = 0
        sigAmean = np.mean(samples[:,count])
        sigAerr = np.sqrt(np.sum((samples[:,count]-np.mean(samples[:,count]))*(samples[:,count]-np.mean(samples[:,count])))/chain_len)
        count += 1

        residBmean = np.mean(samples[:,count])
        residBerr = np.sqrt(np.sum((samples[:,count]-np.mean(samples[:,count]))*(samples[:,count]-np.mean(samples[:,count])))/chain_len)
        count += 1

        sigBmean = np.mean(samples[:,count])
        sigBerr = np.sqrt(np.sum((samples[:,count]-np.mean(samples[:,count]))*(samples[:,count]-np.mean(samples[:,count])))/chain_len)
        count += 1

        residB2mean = np.mean(samples[:,count])
        residB2err = np.sqrt(np.sum((samples[:,count]-np.mean(samples[:,count]))*(samples[:,count]-np.mean(samples[:,count])))/chain_len)
        count += 1

        sigB2mean = np.mean(samples[:,count])
        sigB2err = np.sqrt(np.sum((samples[:,count]-np.mean(samples[:,count]))*(samples[:,count]-np.mean(samples[:,count])))/chain_len)
        count += 1

        skewBmean = np.mean(samples[:,count])
        skewBerr = np.sqrt(np.sum((samples[:,count]-np.mean(samples[:,count]))*(samples[:,count]-np.mean(samples[:,count])))/chain_len)
        count += 1

        fracBmean = np.mean(samples[:,count])
        fracBerr = np.sqrt(np.sum((samples[:,count]-np.mean(samples[:,count]))*(samples[:,count]-np.mean(samples[:,count])))/chain_len)
        count +=1

        fracB2mean = np.mean(samples[:,count])
        fracB2err = np.sqrt(np.sum((samples[:,count]-np.mean(samples[:,count]))*(samples[:,count]-np.mean(samples[:,count])))/chain_len)
        count +=1

        lstepmean = np.mean(samples[:,count])
        lsteperr = np.sqrt(np.sum((samples[:,count]-np.mean(samples[:,count]))*(samples[:,count]-np.mean(samples[:,count])))/chain_len)
        count += 1

        alphamean = np.mean(samples[:,count])
        alphaerr = np.sqrt(np.sum((samples[:,count]-np.mean(samples[:,count]))*(samples[:,count]-np.mean(samples[:,count])))/chain_len)
        count += 1

        betamean = np.mean(samples[:,count])
        betaerr = np.sqrt(np.sum((samples[:,count]-np.mean(samples[:,count]))*(samples[:,count]-np.mean(samples[:,count])))/chain_len)
        count += 1

        outlinefmt = "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f "
        outlinefmt += "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f"
        for par,zcntrl,i in zip(mupar,zcontrol,range(len(mupar))):
            outline = outlinefmt%(np.mean(samples[:,i+11]),np.sqrt(cov[i,i]),par[1],par[2],
                                  sigAmean,sigAerr,sigA[1],sigA[2],
                                  residBmean,residBerr,residB[1],residB[2],
                                  sigBmean,sigBerr,sigB[1],sigB[2],
                                  residB2mean,residB2err,residB2[1],residB2[2],
                                  sigB2mean,sigB2err,sigB2[1],sigB2[2],
                                  skewBmean,skewBerr,skewB[1],skewB[2],
                                  fracBmean,fracBerr,fracB[1],fracB[2],
                                  fracB2mean,fracB2err,fracB2[1],fracB2[2],
                                  lstepmean,lsteperr,lstep[1],lstep[2],
                                  alphamean,alphaerr,alpha[1],alpha[2],
                                  betamean,betaerr,beta[1],beta[2])
            if self.options.append or not os.path.exists(self.options.outputfile) or self.options.clobber:
                fout = open(self.options.outputfile,'a')
                print >> fout, outline
                fout.close()

            if self.options.verbose:
                print("""muA: %.3f +/- %.3f muB: %.3f +/- %.3f sigB: %.3f +/- %.3f muB2: %.3f +/- %.3f sigB2: %.3f +/- %.3f 
skew B: %.3f +/- %.3f frac. B: %.3f +/- %.3f frac. B2: %.3f +/- %.3f Lstep: %.3f +/- %.3f alpha: %.3f +/- %.3f beta: %.3f +/- %.3f"""%(
                        np.mean(samples[:,i+11]),np.sqrt(cov[i,i]),
                        residBmean,residBerr,
                        sigBmean,sigBerr,
                        residB2mean,residB2err,
                        sigB2mean,sigB2err,
                        skewBmean,skewBerr,
                        fracBmean,fracBerr,
                        fracB2mean,fracB2err,
                        lstepmean,lsteperr,
                        alphamean,alphaerr,
                        betamean,betaerr))

    def mcmc(self,inp,zcontrol):
        from scipy.optimize import minimize
        import emcee
        if not inp.__dict__.has_key('PL'):
            inp.PL = 0

        if self.options.fracBguess > 0:
            omitfracB = False
        else:
            omitfracB = True
        # minimize, not maximize
        if self.options.twogauss:
            lnlikefunc = lambda *args: -threegausslike(*args)
        elif self.options.skewedgauss:
            lnlikefunc = lambda *args: -twogausslike_skew(*args)
        elif self.options.simcc:
            lnlikefunc = lambda *args: -twogausslike_simcc(*args)
        else:
            lnlikefunc = lambda *args: -twogausslike(*args)

        inp.mu,inp.muerr = salt2mu_aberr(x1=inp.x1,x1err=inp.x1ERR,c=inp.c,cerr=inp.cERR,mb=inp.mB,mberr=inp.mBERR,
                                         cov_x1_c=inp.COV_x1_c,cov_x1_x0=inp.COV_x1_x0,cov_c_x0=inp.COV_c_x0,
                                         alpha=self.options.salt2alpha,alphaerr=self.options.salt2alphaerr,
                                         beta=self.options.salt2beta,betaerr=self.options.salt2betaerr,
                                         x0=inp.x0,z=inp.zHD)
        inp.residerr = inp.muerr[:]
        inp.resid = inp.mu - cosmo.distmod(inp.zHD).value
        guess = (self.options.popAguess[1],
                 self.options.popBguess[0],self.options.popBguess[1],
                 self.options.popB2guess[0],self.options.popB2guess[1],
                 self.options.skewBguess,
                 self.options.fracBguess,self.options.fracB2guess,
                 self.options.lstepguess,self.options.salt2alphaguess,
                 self.options.salt2betaguess)
        for z in zcontrol:
            guess += (self.options.popAguess[0]+cosmo.distmod(z).value,)
        md = minimize(lnlikefunc,guess,
                      args=(inp,zcontrol,False,False))
        if md.message != 'Optimization terminated successfully.':
            print("""Warning : Minimization Failed!!!  
Try some different initial guesses, or let the MCMC try and take care of it""")


        # for the fixed parameters, make really narrow priors
        md = self.fixedpriors(md)
        md["x"][9] = self.options.salt2alphaguess
        md["x"][10] = self.options.salt2betaguess

        ndim, nwalkers = len(md["x"]), int(self.options.nwalkers)
        pos = [md["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
                                        args=(inp,zcontrol,
                                              omitfracB,self.options.snpars,
                                              self.options.twogauss,self.options.skewedgauss,
                                              self.options.simcc,
                                              self.options.p_residA,self.options.psig_residA,
                                              self.options.p_residB,self.options.psig_residB,
                                              self.options.p_residB2,self.options.psig_residB2,
                                              self.options.p_sigA,self.options.psig_sigA,
                                              self.options.p_sigB,self.options.psig_sigB,
                                              self.options.p_sigB2,self.options.psig_sigB2,
                                              self.options.p_skewB,self.options.psig_skewB,
                                              self.options.p_fracB,self.options.psig_fracB,
                                              self.options.p_fracB2,self.options.psig_fracB2,
                                              self.options.p_lstep,self.options.psig_lstep,
                                              self.options.p_salt2alpha,self.options.psig_salt2alpha,
                                              self.options.p_salt2beta,self.options.psig_salt2beta),
                                        threads=int(self.options.nthreads))
        pos, prob, state = sampler.run_mcmc(pos, 200)
        sampler.reset()
        sampler.run_mcmc(pos, self.options.nsteps, thin=1)
        samples = sampler.flatchain

        # get the error bars - should really return the full posterior
        params = \
            map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                zip(*np.percentile(samples, [16, 50, 84],
                                   axis=0)))

        cov = covmat(samples[:,np.shape(samples)[1] - self.options.nzbins:])
        return(cov,params,samples)

    def fixedpriors(self,md):

        if self.options.popAfixed[0]:
            self.options.p_residA = self.options.popAguess[0]
            self.options.psig_residA = 1e-3
            md["x"][0] = self.options.popAguess[0]
        if self.options.popAfixed[1]:
            self.options.p_sigA = self.options.popAguess[1]
            self.options.psig_sigA = 1e-3
            md["x"][1] = self.options.popAguess[1]
        if self.options.popBfixed[0]:
            self.options.p_residB = self.options.popBguess[0]
            self.options.psig_residB = 1e-3
            md["x"][2] = self.options.popBguess[0]
        if self.options.popBfixed[1]:
            self.options.p_sigB = self.options.popBguess[1]
            self.options.psig_sigB = 1e-3
            md["x"][3] = self.options.popBguess[1]
        if self.options.popB2fixed[0]:
            self.options.p_residB2 = self.options.popB2guess[0]
            self.options.psig_residB2 = 1e-3
            md["x"][4] = self.options.popBguess[0]
        if self.options.popB2fixed[1]:
            self.options.p_sigB2 = self.options.popB2guess[1]
            self.options.psig_sigB2 = 1e-3
            md["x"][5] = self.options.popB2guess[1]
        if self.options.skewBfixed:
            self.options.p_skewB = self.options.skewBguess
            self.options.psig_skewB = 1e-3
            md["x"][6] = self.options.skewBguess
        if self.options.fracBfixed:
            self.options.p_fracB = self.options.fracBguess[0]
            self.options.psig_fracB = 1e-3
            md["x"][7] = self.options.fracBguess[0]
        if self.options.fracB2fixed:
            self.options.p_fracB2 = self.options.fracB2guess[0]
            self.options.psig_fracB2 = 1e-3
            md["x"][8] = self.options.fracB2guess[0]
        if self.options.lstepfixed:
            self.options.p_lstep = self.options.lstepguess
            self.options.psig_lstep = 1e-3
            md["x"][9] = self.options.lstepguess
        if self.options.salt2alphafixed:
            self.options.p_salt2alpha = self.options.salt2alphaguess
            self.options.psig_salt2alpha = 1e-3
            md["x"][10] = self.options.salt2alpha
        if self.options.salt2betafixed:
            self.options.p_salt2beta = self.options.salt2betaguess
            self.options.psig_salt2beta = 1e-3
            md["x"][10] = self.options.salt2beta

        if not self.options.twogauss:
            self.options.p_residB2 = None
            self.options.p_sigB2 = None
            self.options.p_fracB2 = None
        if not self.options.skewedgauss:
            self.options.p_skewB = None

        return(md)

    def transformOptions(self):

        self.options.p_residA = self.options.popAprior_mean[0]
        self.options.psig_residA = self.options.popAprior_mean[1]
        self.options.p_residB = self.options.popBprior_mean[0]
        self.options.psig_residB = self.options.popBprior_mean[1]
        self.options.p_sigA = self.options.popAprior_std[0]
        self.options.psig_sigA = self.options.popAprior_std[1]
        self.options.p_sigB = self.options.popBprior_std[0]
        self.options.psig_sigB = self.options.popBprior_std[1]
        self.options.p_fracB = self.options.fracBprior[0]
        self.options.psig_fracB = self.options.fracBprior[1]

        self.options.p_residB2 = self.options.popB2prior_mean[0]
        self.options.psig_residB2 = self.options.popB2prior_mean[1]
        self.options.p_sigB2 = self.options.popB2prior_std[0]
        self.options.psig_sigB2 = self.options.popB2prior_std[1]
        self.options.p_skewB = self.options.skewBprior[0]
        self.options.psig_skewB = self.options.skewBprior[1]
        self.options.p_fracB2 = self.options.fracB2prior[0]
        self.options.psig_fracB2 = self.options.fracB2prior[1]
        self.options.p_lstep = self.options.lstepprior[0]
        self.options.psig_lstep = self.options.lstepprior[1]

        self.options.p_salt2alpha = self.options.salt2alphaprior[0]
        self.options.psig_salt2alpha = self.options.salt2alphaprior[1]
        self.options.p_salt2beta = self.options.salt2betaprior[0]
        self.options.psig_salt2beta = self.options.salt2betaprior[1]

def twogausslike(x,inp=None,zcontrol=None,frac=True,snpars=True):

    if snpars:
        muA,muAerr = salt2mu(x1=inp.x1,x1err=inp.x1ERR,c=inp.c,cerr=inp.cERR,mb=inp.mB,mberr=inp.mBERR,
                             cov_x1_c=inp.COV_x1_c,cov_x1_x0=inp.COV_x1_x0,cov_c_x0=inp.COV_c_x0,
                             alpha=x[9],beta=x[10],
                             x0=inp.x0,z=inp.zHD)
    else: muA,muAerr = inp.mu[:],inp.muerr[:]

    if frac: fracIa = 1-x[6]; fracCC = x[6]
    else: fracIa = 1; fracCC = 1

    muB,muBerr = inp.mu[:],inp.muerr[:]
    mumodel = np.zeros(len(inp.zHD))
    for mub,mub1,zb,zb1 in zip(x[11:-1],x[12:],zcontrol[:-1],zcontrol[1:]):
        cols = np.where((inp.zHD >= zb) & (inp.zHD < zb1))[0]
        alpha = np.log10(inp.zHD[cols]/zb)/np.log10(zb1/zb)
        mumodel[cols] = (1-alpha)*mub + alpha*mub1

    lnlike = np.sum(logsumexp([-(muA-mumodel)**2./(2.0*(muAerr**2.+x[0]**2.)) + \
                                    np.log((1-x[6])*inp.PA*(1-inp.PL)/(np.sqrt(2*np.pi)*np.sqrt(x[0]**2.+muAerr**2.))),
                                -(muA-mumodel+x[8])**2./(2.0*(muAerr**2.+x[0]**2.)) + \
                                    np.log((1-x[6])*inp.PA*inp.PL/(np.sqrt(2*np.pi)*np.sqrt(x[0]**2.+muAerr**2.))),
                                -(muB-mumodel-x[1])**2./(2.0*(muBerr**2.+x[2]**2.)) + \
                                    np.log((x[6])*(1-inp.PA)/(np.sqrt(2*np.pi)*np.sqrt(x[2]**2.+muBerr**2.)))],axis=0))

    return(lnlike)

def threegausslike(x,inp=None,zcontrol=None,frac=True,snpars=True):

    if snpars:
        muA,muAerr = salt2mu(x1=inp.x1,x1err=inp.x1ERR,c=inp.c,cerr=inp.cERR,mb=inp.mB,mberr=inp.mBERR,
                             cov_x1_c=inp.COV_x1_c,cov_x1_x0=inp.COV_x1_x0,cov_c_x0=inp.COV_c_x0,
                             alpha=x[9],beta=x[10],
                             x0=inp.x0,z=inp.zHD)
    else:
        muA,muAerr = inp.mu[:],inp.muerr[:]
    muB,muBerr = inp.mu[:],inp.muerr[:]

    mumodel = np.zeros(len(inp.zHD))
    for mub,mub1,zb,zb1 in zip(x[11:-1],x[12:],zcontrol[:-1],zcontrol[1:]):
        cols = np.where((inp.zHD >= zb) & (inp.zHD < zb1))[0]
        alpha = np.log10(inp.zHD[cols]/zb)/np.log10(zb1/zb)
        mumodel[cols] = (1-alpha)*mub + alpha*mub1

    if frac: fracIa = 1-x[6]; fracCCA = x[6]; fracCCB = x[7]
    else: fracIa = 1; fracCCA = 1; fracCCB = 1

    sum = logsumexp([-(muA-mumodel)**2./(2.0*(muAerr**2.+x[0]**2.)) + \
                          np.log((1-x[6]-x[7])*inp.PA*(1-inp.PL)/(np.sqrt(2*np.pi)*np.sqrt(x[0]**2.+muAerr**2.))),
                      -(muA-mumodel-x[8])**2./(2.0*(muAerr**2.+x[0]**2.)) + \
                          np.log((1-x[6]-x[7])*inp.PA*inp.PL/(np.sqrt(2*np.pi)*np.sqrt(x[0]**2.+muAerr**2.))),
                      -(muB-mumodel-x[1])**2./(2.0*(muBerr**2.+x[2]**2.)) + \
                          np.log((x[6])*(1-inp.PA)/(np.sqrt(2*np.pi)*np.sqrt(x[2]**2.+muBerr**2.))),
                      -(muB-mumodel-x[3])**2./(2.0*(muBerr**2.+x[4]**2.)) + \
                          np.log((x[7])*(1-inp.PA)/(np.sqrt(2*np.pi)*np.sqrt(x[4]**2.+muBerr**2.)))],axis=0)

    return np.sum(sum)

def twogausslike_skew(x,inp=None,zcontrol=None,frac=True,snpars=True):

    if snpars:
        muA,muAerr = salt2mu(x1=inp.x1,x1err=inp.x1ERR,c=inp.c,cerr=inp.cERR,mb=inp.mB,mberr=inp.mBERR,
                             cov_x1_c=inp.COV_x1_c,cov_x1_x0=inp.COV_x1_x0,cov_c_x0=inp.COV_c_x0,
                             alpha=x[9],beta=x[10],
                             x0=inp.x0,z=inp.zHD)
    else:
        muA,muAerr = inp.mu[:],inp.muerr[:]
    muB,muBerr = inp.mu[:],inp.muerr[:]

    if frac: fracIa = 1-x[6]; fracCC = x[6]
    else: fracIa = 1; fracCC = 1
    
    mumodel = np.zeros(len(inp.zHD))
    for mub,mub1,zb,zb1 in zip(x[11:-1],x[12:],zcontrol[:-1],zcontrol[1:]):
        cols = np.where((inp.zHD >= zb) & (inp.zHD < zb1))[0]
        alpha = np.log10(inp.zHD[cols]/zb)/np.log10(zb1/zb)
        mumodel[cols] = (1-alpha)*mub + alpha*mub1

    gaussA = -(muA-mumodel)**2./(2.0*(muAerr**2.+x[0]**2.)) + \
        np.log((1-x[6])*(inp.PA)*(1-inp.PL)/(np.sqrt(2*np.pi)*np.sqrt(x[0]**2.+muAerr**2.)))
    gaussAhm = -(muA-mumodel-x[8])**2./(2.0*(muAerr**2.+x[0]**2.)) + \
        np.log((1-x[6])*(inp.PA*inp.PL)/(np.sqrt(2*np.pi)*np.sqrt(x[0]**2.+muAerr**2.)))

    normB = x[6]*(1-inp.PA)/(np.sqrt(2*np.pi)*np.sqrt(x[2]**2.+muBerr**2.))
    gaussB = -(muB-mumodel-x[1])**2./(2*(x[2]**2.+muBerr**2.))
    skewB = 1 + erf(x[5]*(muB-mumodel-x[1])/np.sqrt(2*(x[2]**2.+muBerr**2.)))
    skewgaussB = gaussB + np.log(normB*skewB)

    lnlike = np.sum(logsumexp([gaussA,gaussAhm,skewgaussB],axis=0))

    return(lnlike)

def lnprior(theta,
            twogauss,skewedgauss,simcc,zcntrl=None,
            p_residA=None,psig_residA=None,
            p_residB=None,psig_residB=None,
            p_residB2=None,psig_residB2=None,
            p_sig_A=None,psig_sig_A=None,
            p_sig_B=None,psig_sig_B=None,
            p_sig_B2=None,psig_sig_B2=None,
            p_sig_skewB=None,psig_sig_skewB=None,
            p_fracB=None,psig_fracB=None,
            p_fracB2=None,psig_fracB2=None,
            p_lstep=None,psig_lstep=None,
            p_salt2alpha=None,psig_salt2alpha=None,
            p_salt2beta=None,psig_salt2beta=None):

    sigA,residB,sigB,residB2,sigB2,skewB,fracB,fracB2,lstep,alpha,beta = theta[:11]
    residAlist = theta[11:]

    p_theta = 0.0
    if p_residA or p_residA == 0.0:
        for residA,z in zip(residAlist,zcntrl):
            p_theta += norm.logpdf(residA,p_residA+cosmo.distmod(z).value,psig_residA)
    if p_residB:
        p_theta += norm.logpdf(residB,p_residB,psig_residB)
    if p_residB2:
        p_theta += norm.logpdf(residB2,p_residB2,psig_residB2)
    if p_sig_A:
        p_theta += norm.logpdf(sigA,p_sig_A,psig_sig_A)
    if p_sig_B:
        p_theta += norm.logpdf(sigB,p_sig_B,psig_sig_B)
    if p_sig_B2:
        p_theta += norm.logpdf(sigB2,p_sig_B2,psig_sig_B2)
    if p_sig_skewB:
        p_theta += norm.logpdf(skewB,p_sig_skewB,psig_sig_skewB)
    if p_fracB:
        p_theta += norm.logpdf(fracB,p_fracB,psig_fracB)
    if p_fracB2:
        p_theta += norm.logpdf(fracB2,p_fracB2,psig_fracB2)
    if type(p_lstep) != None:
        p_lstep += norm.logpdf(lstep,p_lstep,psig_lstep)
    if p_salt2alpha:
        p_theta += norm.logpdf(alpha,p_salt2alpha,psig_salt2alpha)
    if p_salt2beta:
        p_theta += norm.logpdf(beta,p_salt2beta,psig_salt2beta)

    if fracB > 1 or fracB < 0: return -np.inf
    elif fracB2 > 1 or fracB2 < 0: return -np.inf
    else: return(p_theta)

def lnprob(theta,inp=None,zcontrol=None,
           omitfracB=False,snpars=False,
           twogauss=False,
           skewedgauss=False,
           simcc=None,
           p_residA=None,psig_residA=None,
           p_residB=None,psig_residB=None,
           p_residB2=None,psig_residB2=None,
           p_sig_A=None,psig_sig_A=None,
           p_sig_B=None,psig_sig_B=None,
           p_sig_B2=None,psig_sig_B2=None,
           p_sig_skewB=None,psig_sig_skewB=None,
           p_fracB=None,psig_fracB=None,
           p_fracB2=None,psig_fracB2=None,
           p_lstep=None,psig_lstep=None,
           p_salt2alpha=None,psig_salt2alpha=None,
           p_salt2beta=None,psig_salt2beta=None):

    if twogauss:
        lnlikefunc = lambda *args: threegausslike(*args)
    elif skewedgauss:
        lnlikefunc = lambda *args: twogausslike_skew(*args)
    elif simcc:
        lnlikefunc = lambda *args: twogausslike_simcc(*args)
    else:
        lnlikefunc = lambda *args: twogausslike(*args)

    lp = lnprior(theta,twogauss,skewedgauss,simcc,zcontrol,
                 p_residA=p_residA,psig_residA=psig_residA,
                 p_residB=p_residB,psig_residB=psig_residB,
                 p_residB2=p_residB2,psig_residB2=psig_residB2,
                 p_sig_A=p_sig_A,psig_sig_A=psig_sig_A,
                 p_sig_B=p_sig_B,psig_sig_B=psig_sig_B,
                 p_sig_B2=p_sig_B2,psig_sig_B2=psig_sig_B2,
                 p_sig_skewB=p_sig_skewB,psig_sig_skewB=psig_sig_skewB,
                 p_fracB=p_fracB,psig_fracB=psig_fracB,
                 p_fracB2=p_fracB2,psig_fracB2=psig_fracB2,
                 p_lstep=p_lstep,psig_lstep=psig_lstep,
                 p_salt2alpha=p_salt2alpha,psig_salt2alpha=psig_salt2alpha,
                 p_salt2beta=p_salt2beta,psig_salt2beta=psig_salt2beta)

    if not np.isfinite(lp) or np.isnan(lp):
        return -np.inf

    post = lp+lnlikefunc(theta,inp,zcontrol,omitfracB,snpars)

    if post != post: return -np.inf
    else: return post


def gauss(x,x0,sigma):
    return(normpdf(x,x0,sigma))

def normpdf(x, mu, sigma):
    u = (x-mu)/np.abs(sigma)
    y = (1/(np.sqrt(2*np.pi)*np.abs(sigma)))*np.exp(-u*u/2)
    return y

def covmat(samples):
    cov_shape = np.shape(samples)[1]
    chain_len = np.shape(samples)[0]
    covmat = np.zeros([cov_shape,cov_shape])
    for i in range(cov_shape):
        for j in range(cov_shape):
            covmat[j,i] = np.sum((samples[:,j]-np.mean(samples[:,j]))*(samples[:,i]-np.mean(samples[:,i])))/chain_len
    return(covmat)

def weighted_avg_and_std(values, weights):
    import numpy
    average = numpy.average(values, weights=weights)
    variance = numpy.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, numpy.sqrt(variance))

def salt2mu(x1=None,x1err=None,
            c=None,cerr=None,
            mb=None,mberr=None,
            cov_x1_c=None,cov_x1_x0=None,cov_c_x0=None,
            alpha=None,beta=None,
            M=None,x0=None,sigint=None,z=None,peczerr=0.0005):

    sf = -2.5/(x0*np.log(10.0))
    cov_mb_c = cov_c_x0*sf
    cov_mb_x1 = cov_x1_x0*sf
    mu_out = mb + x1*alpha - beta*c + 19.3
    invvars = 1.0 / (mberr**2.+ alpha**2. * x1err**2. + beta**2. * cerr**2. + \
                         2.0 * alpha * (cov_x1_x0*sf) - 2.0 * beta * (cov_c_x0*sf) - \
                         2.0 * alpha*beta * (cov_x1_c) )

    zerr = peczerr*5.0/np.log(10)*(1.0+z)/(z*(1.0+z/2.0))
    muerr_out = np.sqrt(1/invvars + zerr**2. + 0.055**2.*z**2.)
    if sigint: muerr_out = np.sqrt(muerr_out**2. + sigint**2.)
    return(mu_out,muerr_out)

def salt2mu_aberr(x1=None,x1err=None,
                  c=None,cerr=None,
                  mb=None,mberr=None,
                  cov_x1_c=None,cov_x1_x0=None,cov_c_x0=None,
                  alpha=None,beta=None,
                  alphaerr=None,betaerr=None,
                  M=None,x0=None,sigint=None,z=None,peczerr=0.0005):
    from uncertainties import ufloat, correlated_values, correlated_values_norm
    alphatmp,betatmp = alpha,beta
    alpha,beta = ufloat(alpha,alphaerr),ufloat(beta,betaerr)

    sf = -2.5/(x0*np.log(10.0))
    cov_mb_c = cov_c_x0*sf
    cov_mb_x1 = cov_x1_x0*sf

    mu_out,muerr_out = np.array([]),np.array([])
    for i in range(len(x1)):

        covmat = np.array([[mberr[i]**2.,cov_mb_x1[i],cov_mb_c[i]],
                           [cov_mb_x1[i],x1err[i]**2.,cov_x1_c[i]],
                           [cov_mb_c[i],cov_x1_c[i],cerr[i]**2.]])
        mb_single,x1_single,c_single = correlated_values([mb[i],x1[i],c[i]],covmat)

        mu = mb_single + x1_single*alpha - beta*c_single + 19.3
        if sigint: mu = mu + ufloat(0,sigint)
        zerr = peczerr*5.0/np.log(10)*(1.0+z[i])/(z[i]*(1.0+z[i]/2.0))

        mu = mu + ufloat(0,np.sqrt(zerr**2. + 0.055**2.*z[i]**2.))
        mu_out,muerr_out = np.append(mu_out,mu.n),np.append(muerr_out,mu.std_dev)

    return(mu_out,muerr_out)

if __name__ == "__main__":
    usagestring="""An implementation of the BEAMS method (Kunz, Bassett, & Hlozek 2006).
Uses Bayesian methods to estimate the mags of a sample with contaminants.  This version
follows Betoule et al. (2014) and sets a series of distance modulus control points, 
simultaneously finding the distance moduli at each of these points via MCMC, and prints 
the full covariance matrix.

Takes a
parameter file or command line options, and a file with the following header/columns:

# PA z mu mu_err
<PA_1> <z> <resid_1> <resid_err_1>
<PA_2> <z> <resid_2> <resid_err_2>
<PA_3> <z> <resid_3> <resid_err_3>
.......

The PA column is the prior probability that a data point belongs to population A, P(A).
Column 2 is some sort of de-trended magnitude measurement (i.e. Hubble residuals), and
column 3 is the uncertainties on those measurements.

Can specify or fix, with priors:

1. The mean and standard deviation of the population of interest
2. The mean and standard deviation of the contaminant population
3. The fraction of contaminants

USAGE: doSNBEAMS.py -p param_file -i input_file [options]

examples:
"""

    import exceptions
    import os
    import optparse
    import ConfigParser

    beam = BEAMS()

    # read in the options from the param file and the command line
    # some convoluted syntax here, making it so param file is not required
    parser = beam.add_options(usage=usagestring)
    options,  args = parser.parse_args()
    if options.paramfile:
        config = ConfigParser.ConfigParser()
        config.read(options.paramfile)
    else: config=None
    parser = beam.add_options(usage=usagestring,config=config)
    options,  args = parser.parse_args()


    beam.options = options
    beam.verbose = options.verbose
    beam.clobber = options.clobber

    beam.transformOptions()

    from scipy.optimize import minimize
    import emcee

    beam.main(options.inputfile)

