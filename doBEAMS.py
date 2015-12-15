#!/usr/bin/env python
# D. Jones - 9/1/15
"""BEAMS method for PS1 data"""
import numpy as np
from scipy.misc import logsumexp
from scipy.special import erf

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

        if config:
            parser.add_option('--clobber', default=map(int,config.get('all','clobber'))[0], action="store_true",
                              help='clobber output file')
            parser.add_option('--append', default=map(int,config.get('all','append'))[0], action="store_true",
                              help='open output file in append mode')

            # Input file
            parser.add_option('--pacol', default=config.get('all','pacol'), type="string",
                              help='column in input file used as prior P(A)')
            parser.add_option('--residcol', default=config.get('all','residcol'), type="string",
                              help='column name in input file header for residuals')
            parser.add_option('--residerrcol', default=config.get('all','residerrcol'), type="string",
                              help='column name in input file header for residual errors')

            # population A guesses and priors (the pop we care about)
            parser.add_option('--popAguess', default=map(float,config.get('all','popAguess').split(',')),type='float',
                              help='comma-separated initial guesses for population A: mean, standard deviation',nargs=2)
            parser.add_option('--popAprior_mean', default=map(float,config.get('all','popAprior_mean').split(',')),type='float',
                              help="""comma-separated gaussian prior for mean of population A: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--popAprior_std', default=map(float,config.get('all','popAprior_std').split(',')),type='float',
                              help="""comma-separated gaussian prior for std dev. of population A: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--popAfixed', default=map(float,config.get('all','popAfixed').split(',')),type='float',
                              help="""comma-separated values for population A params: mean, standard deviation.  
For each param, set to 0 to include in parameter estimation, set to 1 to keep fixed""",nargs=2)


            # population B guesses and priors (the pop we care about)
            parser.add_option('--popBguess', default=map(float,config.get('all','popBguess').split(',')),type='float',
                              help='comma-separated initial guesses for population B: mean, standard deviation',nargs=2)
            parser.add_option('--popBprior_mean', default=map(float,config.get('all','popBprior_mean').split(',')),type='float',
                              help="""comma-separated gaussian prior for mean of population B: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--popBprior_std', default=map(float,config.get('all','popBprior_std').split(',')),type='float',
                              help="""comma-separated gaussian prior for std dev. of population B: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--popBfixed', default=map(float,config.get('all','popBfixed').split(',')),type='float',
                              help="""comma-separated values for population B params: mean, standard deviation.  
For each param, set to 0 to include in parameter estimation, set to 1 to keep fixed""",nargs=2)


            # population B guesses and priors for second gaussian (the pop we care about)
            parser.add_option('--popB2guess', default=map(float,config.get('all','popB2guess').split(',')),type='float',
                              help='comma-separated initial guesses for population B: mean, standard deviation',nargs=2)
            parser.add_option('--popB2prior_mean', default=map(float,config.get('all','popB2prior_mean').split(',')),type='float',
                              help="""comma-separated gaussian prior for mean of population B: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--popB2prior_std', default=map(float,config.get('all','popB2prior_std').split(',')),type='float',
                              help="""comma-separated gaussian prior for std dev. of population B: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--popB2fixed', default=map(float,config.get('all','popB2fixed').split(',')),type='float',
                              help="""comma-separated values for population B params: mean, standard deviation.  
For each param, set to 0 to include in parameter estimation, set to 1 to keep fixed""",nargs=2)

            # the skewness of the pop. B gaussian
            parser.add_option('--skewBguess', default=config.get('all','skewBguess'),type='float',
                              help='comma-separated initial guesses for population B: mean, standard deviation')
            parser.add_option('--skewBprior', default=map(float,config.get('all','skewBprior').split(',')),type='float',
                              help="""comma-separated gaussian prior for skewness of population B: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--skewBfixed', default=config.get('all','skewBfixed'),type='int',
                              help="""set to 0 to include skewness in parameter estimation, set to 1 to keep fixed""")

            # fraction of contaminants: guesses and priors
            parser.add_option('--fracBguess', default=config.get('all','fracBguess'),type='float',
                              help='initial guess for fraction of contaminants, set to negative to omit')
            parser.add_option('--fracBprior', default=map(float,config.get('all','fracBprior').split(',')),type='float',
                              help="""comma-separated gaussian prior on fraction of contaminants: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--fracBfixed', default=config.get('all','fracBfixed'),type='int',
                              help="""0 to return posterior frac. of contaminants, 1 to keep fixed""")

            # fraction of contaminants: guesses and priors for gaussian 2
            parser.add_option('--fracB2guess', default=config.get('all','fracBguess'),type='float',
                              help='initial guess for fraction of contaminants, set to negative to omit')
            parser.add_option('--fracB2prior', default=map(float,config.get('all','fracBprior').split(',')),type='float',
                              help="""comma-separated gaussian prior on fraction of contaminants: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--fracB2fixed', default=config.get('all','fracBfixed'),type='int',
                              help="""0 to return posterior frac. of contaminants, 1 to keep fixed""")

            # options to fit to a second-order "step" effect on luminosity.
            # This is the host mass bias for SNe.
            parser.add_option('--plcol', default=config.get('all','plcol'), type="string",
                              help='column in input file used as probability for an additional luminosity correction P(L)')
            parser.add_option('--lstep', default=config.get('all','lstep'), type="int",
                              help='make step correction if set to 1 (default=%default)')
            parser.add_option('--lstepguess', default=config.get('all','lstepguess'),type='float',
                              help='initial guesses for luminosity correction')
            parser.add_option('--lstepprior', default=map(float,config.get('all','lstepprior').split(',')),type='float',
                              help="""comma-separated gaussian prior for mean of luminosity correction: centroid, sigma.  
For flat prior, use empty strings""",nargs=2)
            parser.add_option('--lstepfixed', default=config.get('all','lstepfixed'),type='float',
                              help="""comma-separated values for luminosity correction.
For each param, set to 0 to include in parameter estimation, set to 1 to keep fixed""")

        
            # output and number of threads
            parser.add_option('--nthreads', default=config.get('all','nthreads'), type="int",
                              help='Number of threads for MCMC')
            parser.add_option('--nwalkers', default=config.get('all','nwalkers'), type="int",
                              help='Number of walkers for MCMC')
            parser.add_option('--nsteps', default=config.get('all','nsteps'), type="int",
                              help='Number of steps for MCMC')
            parser.add_option('--ninit', default=config.get('all','ninit'), type="int",
                              help="Number of steps before the samples wander away from the initial values and are 'burnt in'")


            parser.add_option('-i','--inputfile', default=config.get('all','inputfile'), type="string",
                              help='file with the input data')
            parser.add_option('-o','--outputfile', default=config.get('all','outputfile'), type="string",
                              help='Output file with the derived parameters for each redshift bin')

            # alternate functional models
            parser.add_option('--twogauss', default=map(int,config.get('all','twogauss'))[0], action="store_true",
                              help='two gaussians for pop. B')
            parser.add_option('--skewedgauss', default=map(int,config.get('all','skewedgauss'))[0], action="store_true",
                              help='skewed gaussian for pop. B')


        else:
            parser.add_option('--clobber', default=False, action="store_true",
                              help='clobber output file')
            parser.add_option('--append', default=False, action="store_true",
                              help='open output file in append mode')

            # Input file
            parser.add_option('--pacol', default='PA', type="string",
                              help='column in input file used as prior P(A)')
            parser.add_option('--residcol', default='resid', type="string",
                              help='column name in input file header for residuals')
            parser.add_option('--residerrcol', default='resid_err', type="string",
                              help='column name in input file header for residual errors')

            # population A guesses and priors (the pop we care about)
            parser.add_option('--popAguess', default=(0.0,0.1),type='float',
                              help='comma-separated initial guesses for population A: mean, standard deviation',nargs=2)
            parser.add_option('--popAprior_mean', default=(0.0,0.1),type='float',
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
            parser.add_option('--fracBprior', default=(0.05,0.1),type='float',
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
            
        
            # output and number of threads
            parser.add_option('--nthreads', default=8, type="int",
                              help='Number of threads for MCMC')
            parser.add_option('--nwalkers', default=100, type="int",
                              help='Number of walkers for MCMC')
            parser.add_option('--nsteps', default=500, type="int",
                              help='Number of steps for MCMC')
            parser.add_option('--ninit', default=50, type="int",
                              help="Number of steps before the samples wander away from the initial values and are 'burnt in'")


            # alternate functional models
            parser.add_option('--twogauss', default=False, action="store_true",
                              help='two gaussians for pop. B')
            parser.add_option('--skewedgauss', default=False, action="store_true",
                              help='skewed gaussian for pop. B')


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
        if self.options.lstep:
            inp.PL = inp.__dict__[self.options.plcol]
        else:
            self.options.lstepfixed = 1
            self.options.lstepguess = 0.0
        inp.resid = inp.__dict__[self.options.residcol]
        inp.residerr = inp.__dict__[self.options.residerrcol]
        
        data = True
        if not len(inp.resid):
            print('Warning : no data in input file!!')
            data = False

        # open the output file
        if not os.path.exists(self.options.outputfile) or self.options.clobber:
            writeout = True
            fout = open(self.options.outputfile,'w')
            outline = "# muA muAerr muAerr_m muAerr_p sigA sigAerr sigAerr_m sigAerr_p muB muBerr muBerr_m muBerr_p "
            outline += "sigB sigBerr sigBerr_m sigBerr_p muB2 muB2err muB2err_m muB2err_p sigB2 sigB2err sigB2err_m "
            outline += "sigB2err_p skewB skewBerr skewBerr_m skewBerr_p fracB fracBerr fracBerr_m fracBerr_p fracB2 "
            outline += "fracB2err fracB2err_m fracB2err_p lstep lsteperr lsteperr_m lsteperr_p"
            print >> fout, outline
            fout.close()
        elif not self.options.append:
            writeout = False
            print('Warning : files %s exists!!  Not clobbering'%self.options.outputfile)

        # run the MCMC
        if data:
            residA,sigA,residB,sigB,residB2,sigB2,skewB,fracB,fracB2,lstep,samples = self.mcmc(inp)
                
            outlinefmt = "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f "
            outlinefmt += "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f"

            chain_len = len(samples[:,0])

            count = 0
            residAmean = np.mean(samples[:,count])
            residAerr = np.sqrt(np.sum((samples[:,count]-np.mean(samples[:,count]))*(samples[:,count]-np.mean(samples[:,count])))/chain_len)
            count += 1

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

            if residAerr < 0.001: residAerr = 1e7#; residA[1] = 1e7; residA[2] = 1e7
            if residBerr < 0.001: residBerr = 1e7#; residB[1] = 1e7; residB[2] = 1e7

            outline = outlinefmt%(residAmean,residAerr,residA[1],residA[2],
                                  sigAmean,sigAerr,sigA[1],sigA[2],
                                  residBmean,residBerr,residB[1],residB[2],
                                  sigBmean,sigBerr,sigB[1],sigB[2],
                                  residB2mean,residB2err,residB2[1],residB2[2],
                                  sigB2mean,sigB2err,sigB2[1],sigB2[2],
                                  skewBmean,skewBerr,skewB[1],skewB[2],
                                  fracBmean,fracBerr,fracB[1],fracB[2],
                                  fracB2mean,fracB2err,fracB2[1],fracB2[2],
                                  lstepmean,lsteperr,lstep[1],lstep[2])
            if self.options.append or not os.path.exists(self.options.outputfile) or self.options.clobber:
                fout = open(self.options.outputfile,'a')
                print >> fout, outline
                fout.close()

            if self.options.verbose:
                print("""muA: %.3f +/- %.3f muB: %.3f +/- %.3f sigB: %.3f +/- %.3f muB2: %.3f +/- %.3f sigB2: %.3f +/- %.3f 
skew B: %.3f +/- %.3f frac. B: %.3f +/- %.3f frac. B2: %.3f +/- %.3f Lstep: %.3f +/- %.3f"""%(
                        residAmean,residAerr,
                        residBmean,residBerr,
                        sigBmean,sigBerr,
                        residB2mean,residB2err,
                        sigB2mean,sigB2err,
                        skewBmean,skewBerr,
                        fracBmean,fracBerr,
                        fracB2mean,fracB2err,
                        lstepmean,lsteperr))

        else:
            outlinefmt = "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f "
            outlinefmt += "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f"

            outline = outlinefmt%(0,1e7,1e7,1e7,
                                  0,1e7,1e7,1e7,
                                  0,1e7,1e7,1e7,
                                  0,1e7,1e7,1e7,
                                  0,1e7,1e7,1e7,
                                  0,1e7,1e7,1e7,
                                  0,1e7,1e7,1e7,
                                  0,1e7,1e7,1e7,
                                  0,1e7,1e7,1e7,
                                  0,1e7,1e7,1e7,
                                  0,1e7,1e7,1e7,
                                  0,1e7,1e7,1e7)
            if self.options.append or not os.path.exists(self.options.outputfile) or self.options.clobber:
                fout = open(self.options.outputfile,'a')
                print >> fout, outline
                fout.close()

    def mcmc(self,inp):
        from scipy.optimize import minimize
        import emcee

        if not inp.__dict__.has_key('PL'):
            inp.PL = 0

        if self.options.fracBguess >= 0:
            omitfracB = False
        else:
            omitfracB = True

        # minimize, not maximize
        if self.options.twogauss:
            lnlikefunc = lambda *args: -threegausslike_nofrac(*args)
        elif self.options.skewedgauss:
            lnlikefunc = lambda *args: -twogausslike_skew_nofrac(*args)
        else:
            lnlikefunc = lambda *args: -twogausslike_nofrac(*args)

        md = minimize(lnlikefunc,(self.options.popAguess[0],self.options.popAguess[1],
                                  self.options.popBguess[0],self.options.popBguess[1],
                                  self.options.popB2guess[0],self.options.popB2guess[1],
                                  self.options.skewBguess,
                                  self.options.fracBguess,self.options.fracB2guess,
                                  self.options.lstepguess),
                      args=(inp.PA,inp.PL,inp.resid,inp.residerr))
        if md.message != 'Optimization terminated successfully.':
            print("""Warning : Minimization Failed!!!
Try some different initial guesses, or let the MCMC try and take care of it""")

        # for the fixed parameters, make really narrow priors
        md = self.fixedpriors(md)
        # md["x"][4] = md["x"][2]-1.0

        ndim, nwalkers = len(md["x"]), int(self.options.nwalkers)
        pos = [md["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        print len(np.where(inp.PA == 0)[0])
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                        args=(inp.PA,inp.PL,inp.resid,inp.residerr,omitfracB,
                                              self.options.twogauss,self.options.skewedgauss,
                                              self.options.p_residA,self.options.psig_residA,
                                              self.options.p_residB,self.options.psig_residB,
                                              self.options.p_residB2,self.options.psig_residB2,
                                              self.options.p_sigA,self.options.psig_sigA,
                                              self.options.p_sigB,self.options.psig_sigB,
                                              self.options.p_sigB2,self.options.psig_sigB2,
                                              self.options.p_skewB,self.options.psig_skewB,
                                              self.options.p_fracB,self.options.psig_fracB,
                                              self.options.p_fracB2,self.options.psig_fracB2,
                                              self.options.p_lstep,self.options.psig_lstep),
                                        threads=int(self.options.nthreads))
        pos, prob, state = sampler.run_mcmc(pos, 200)
        sampler.reset()
        sampler.run_mcmc(pos, self.options.nsteps, thin=1)
        samples = sampler.flatchain#[:, self.options.ninit:, :].reshape((-1, ndim))

        # get the error bars - should really return the full posterior!
        import scipy.stats
        resida_mcmc, siga_mcmc, residb_mcmc, sigb_mcmc, residb2_mcmc, sigb2_mcmc, skewB_mcmc, fracB, fracB2, lstep = \
            map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                zip(*np.percentile(samples, [16, 50, 84],
                                   axis=0)))

        return(resida_mcmc,siga_mcmc,residb_mcmc,sigb_mcmc,residb2_mcmc,sigb2_mcmc,skewB_mcmc,fracB,fracB2,lstep,samples)

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

def twogausslike(x,PA=None,PL=None,resid=None,residerr=None):
    
    if type(PL) != None and type(PL) != int:
        return np.sum(np.logaddexp(-(resid-x[0]-x[5])**2./(2.0*(residerr**2.+x[1]**2.)) + \
                                        np.log((1-x[4])*(PA)*PL/(np.sqrt(2*np.pi)*np.sqrt(x[1]**2.+residerr**2.))),
                                    -(resid-x[2])**2./(2.0*(residerr**2.+x[3]**2.)) + \
                                        np.log((x[4])*(1-PA)/(np.sqrt(2*np.pi)*np.sqrt(x[3]**2.+residerr**2.)))) + \
                          np.logaddexp(-(resid-x[0])**2./(2.0*(residerr**2.+x[1]**2.)) + \
                                            np.log((1-x[4])*(PA)*(1-PL)/(np.sqrt(2*np.pi)*np.sqrt(x[1]**2.+residerr**2.))),
                                        0))
    else:
        return np.sum(np.logaddexp(-(resid-x[0])**2./(2.0*(residerr**2.+x[1]**2.)) + \
                                        np.log((1-x[-3])*(PA)/(np.sqrt(2*np.pi)*np.sqrt(x[1]**2.+residerr**2.))),
                                    -(resid-x[0])**2./(2.0*(residerr**2.+x[3]**2.)) + \
                                        np.log((x[-3])*(1-PA)/(np.sqrt(2*np.pi)*np.sqrt(x[3]**2.+residerr**2.)))))


def twogausslike_nofrac(x,PA=None,PL=None,resid=None,residerr=None):

    return np.sum(np.logaddexp(-(resid-x[0])**2./(2.0*(residerr**2.+x[1]**2.)) + \
                                    np.log(PA/(np.sqrt(2*np.pi)*np.sqrt(x[1]**2.+residerr**2.))),
                                -(resid-x[2])**2./(2.0*(residerr**2.+x[3]**2.)) + \
                                    np.log((1-PA)/(np.sqrt(2*np.pi)*np.sqrt(x[3]**2.+residerr**2.)))))

def threegausslike(x,PA=None,PL=None,resid=None,residerr=None):

    sum = logsumexp([-(resid-x[0])**2./(2.0*(residerr**2.+x[1]**2.)) + \
                          np.log((1-x[-3]-x[-2])*(PA)/(np.sqrt(2*np.pi)*np.sqrt(x[1]**2.+residerr**2.))),
                      -(resid-x[2])**2./(2.0*(residerr**2.+x[3]**2.)) + \
                          np.log((x[-3])*(1-PA)/(np.sqrt(2*np.pi)*np.sqrt(x[3]**2.+residerr**2.))),
                      -(resid-x[4])**2./(2.0*(residerr**2.+x[5]**2.)) + \
                          np.log((x[-2])*(1-PA)/(np.sqrt(2*np.pi)*np.sqrt(x[5]**2.+residerr**2.)))],axis=0)
    return np.sum(sum)

def threegausslike_nofrac(x,PA=None,PL=None,resid=None,residerr=None):

    sum = logsumexp([-(resid-x[0])**2./(2.0*(residerr**2.+x[1]**2.)) + \
                          np.log((PA)/(np.sqrt(2*np.pi)*np.sqrt(x[1]**2.+residerr**2.))),
                      -(resid-x[2])**2./(2.0*(residerr**2.+x[3]**2.)) + \
                          np.log((1-PA)/(np.sqrt(2*np.pi)*np.sqrt(x[3]**2.+residerr**2.))),
                      -(resid-x[4])**2./(2.0*(residerr**2.+x[5]**2.)) + \
                          np.log((1-PA)/(np.sqrt(2*np.pi)*np.sqrt(x[5]**2.+residerr**2.)))],axis=0)
    return np.sum(sum)

#    return np.sum(sum)

def twogausslike_skew(x,PA=None,PL=None,resid=None,residerr=None):

    gaussA = -(resid-x[0])**2./(2.0*(residerr**2.+x[1]**2.)) + \
        np.log((1-x[-1])*(PA)/(np.sqrt(2*np.pi)*np.sqrt(x[1]**2.+residerr**2.))),
    normB = x[-1]*(1-PA)/(np.sqrt(2*np.pi)*np.sqrt(x[3]**2.+residerr**2.))
    gaussB = -(resid-x[2])**2./(2*(x[3]**2.+residerr**2.))
    skewB = 1 + erf(x[6]*(resid-x[2])/np.sqrt(2*(x[3]**2.+residerr**2.)))
    skewgaussB = gaussB + np.log(normB*skewB)
    
    return np.sum(np.logaddexp(gaussA,skewgaussB))

def twogausslike_skew_nofrac(x,PA=None,PL=None,resid=None,residerr=None):

    gaussA = -(resid-x[0])**2./(2.0*(residerr**2.+x[1]**2.)) + \
        np.log((PA)/(np.sqrt(2*np.pi)*np.sqrt(x[1]**2.+residerr**2.))),
    normB = (1-PA)/(np.sqrt(2*np.pi)*np.sqrt(x[3]**2.+residerr**2.))
    gaussB = -(resid-x[2])**2./(2*(x[3]**2.+residerr**2.))
    skewB = 1 + erf(x[6]*(resid-x[2])/np.sqrt(2*(x[3]**2.+residerr**2.)))
    skewgaussB = gaussB + np.log(normB*skewB)

    return np.sum(np.logaddexp(gaussA,skewgaussB))


def lnprior(theta,twogauss,skewedgauss,
            p_residA=None,psig_residA=None,
            p_residB=None,psig_residB=None,
            p_residB2=None,psig_residB2=None,
            p_sig_A=None,psig_sig_A=None,
            p_sig_B=None,psig_sig_B=None,
            p_sig_B2=None,psig_sig_B2=None,
            p_sig_skewB=None,psig_sig_skewB=None,
            p_fracB=None,psig_fracB=None,
            p_fracB2=None,psig_fracB2=None,
            p_lstep=None,psig_lstep=None):

    residA,sigA,residB,sigB,residB2,sigB2,skewB,fracB,fracB2,lstep = theta

    p_theta = 1.0

    if p_residA:
        p_theta *= gauss(residA,p_residA,psig_residA)
    if p_residB:
        p_theta *= gauss(residB,p_residB,psig_residB)
    if p_residB2:
        p_theta *= gauss(residB2,p_residB2,psig_residB2)
    if p_sig_A:
        p_theta *= gauss(sigA,p_sig_A,psig_sig_A)
    if p_sig_B:
        p_theta *= gauss(sigB,p_sig_B,psig_sig_B)
    if p_sig_B2:
        p_theta *= gauss(sigB2,p_sig_B2,psig_sig_B2)
    if p_sig_skewB:
        p_theta *= gauss(skewB,p_sig_skewB,psig_sig_skewB)
    if p_fracB:
        p_theta *= gauss(fracB,p_fracB,psig_fracB)
    if p_fracB2:
        p_theta *= gauss(fracB2,p_fracB2,psig_fracB2)
    if type(p_lstep) != None:
        p_lstep *= gauss(lstep,p_lstep,psig_lstep)

    if fracB > 1 or fracB < 0: return -np.inf
    elif fracB2 > 1 or fracB2 < 0: return -np.inf
    else: return(np.log(p_theta))

def lnprob(theta,PA=None,PL=None,resid=None,residerr=None,
           omitfracB=False,twogauss=False,
           skewedgauss=False,
           p_residA=None,psig_residA=None,
           p_residB=None,psig_residB=None,
           p_residB2=None,psig_residB2=None,
           p_sig_A=None,psig_sig_A=None,
           p_sig_B=None,psig_sig_B=None,
           p_sig_B2=None,psig_sig_B2=None,
           p_sig_skewB=None,psig_sig_skewB=None,
           p_fracB=None,psig_fracB=None,
           p_fracB2=None,psig_fracB2=None,
           p_lstep=None,psig_lstep=None):

    if twogauss and omitfracB:
        lnlikefunc = lambda *args: threegausslike_nofrac(*args)
    elif skewedgauss and omitfracB:
        lnlikefunc = lambda *args: twogausslike_skew_nofrac(*args)
    elif not twogauss and not skewedgauss and omitfracB:
        lnlikefunc = lambda *args: twogausslike_nofrac(*args)
    elif twogauss:
        lnlikefunc = lambda *args: threegausslike(*args)
    elif skewedgauss:
        lnlikefunc = lambda *args: twogausslike_skew(*args)
    else:
        lnlikefunc = lambda *args: twogausslike(*args)

    lp = lnprior(theta,twogauss,skewedgauss,
                 p_residA=p_residA,psig_residA=psig_residA,
                 p_residB=p_residB,psig_residB=psig_residB,
                 p_residB2=p_residB2,psig_residB2=psig_residB2,
                 p_sig_A=p_sig_A,psig_sig_A=psig_sig_A,
                 p_sig_B=p_sig_B,psig_sig_B=psig_sig_B,
                 p_sig_B2=p_sig_B2,psig_sig_B2=psig_sig_B2,
                 p_sig_skewB=p_sig_skewB,psig_sig_skewB=psig_sig_skewB,
                 p_fracB=p_fracB,psig_fracB=psig_fracB,
                 p_fracB2=p_fracB2,psig_fracB2=psig_fracB2,
                 p_lstep=p_lstep,psig_lstep=psig_lstep)

    if not np.isfinite(lp) or np.isnan(lp):
        return -np.inf

    post = lp+lnlikefunc(theta,PA,PL,
                         resid,residerr)

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
            covmat[j,i] = np.sum(samples[j]*samples[i])/chain_len
    return(covmat)

if __name__ == "__main__":
    usagestring="""An implementation of the BEAMS method (Kunz, Bassett, & Hlozek 2006).
Uses Bayesian methods to estimate the mags of a sample with contaminants.  Takes a
parameter file or command line options, and a file with the following header/columns:

# PA resid resid_err
<PA_1> <resid_1> <resid_err_1>
<PA_2> <resid_2> <resid_err_2>
<PA_3> <resid_3> <resid_err_3>
.......

The PA column is the prior probability that a data point belongs to population A, P(A).
Column 2 is some sort of de-trended magnitude measurement (i.e. Hubble residuals), and
column 3 is the uncertainties on those measurements.

Can specify or fix, with priors:

1. The mean and standard deviation of the population of interest
2. The mean and standard deviation of the contaminant population
3. The fraction of contaminants

USAGE: doBEAMS.py -p param_file -i input_file [options]

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
