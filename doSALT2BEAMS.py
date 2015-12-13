#!/usr/bin/env python
# D. Jones - 9/1/15
"""BEAMS method for PS1 data"""
import numpy as np

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


            # fraction of contaminants: guesses and priors
            parser.add_option('--fracBguess', default=config.get('all','fracBguess'),type='float',
                              help='initial guess for fraction of contaminants, set to negative to omit')
            parser.add_option('--fracBprior', default=map(float,config.get('all','fracBprior').split(',')),type='float',
                              help="""comma-separated gaussian prior on fraction of contaminants: centroid, sigma.
For flat prior, use empty string""",nargs=2)
            parser.add_option('--fracBfixed', default=config.get('all','fracBfixed'),type='int',
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

            # SALT2 SN parameters: guesses and priors
            parser.add_option('--salt2alphaguess', default=config.get('all','salt2alphaguess'),type='float',
                              help='initial guess for fraction of contaminants, set to negative to omit')
            parser.add_option('--salt2alphaprior',
                              default=map(float,config.get('all','salt2alphaprior').split(',')),type='float',
                              help="""comma-separated gaussian prior on fraction of contaminants: centroid, sigma.
For flat prior, use empty string""",nargs=2)
            parser.add_option('--salt2alphafixed', default=config.get('all','salt2alphafixed'),type='int',
                              help="""0 to return posterior frac. of contaminants, 1 to keep fixed""")
            parser.add_option('--salt2betaguess', default=config.get('all','salt2betaguess'),type='float',
                              help='initial guess for fraction of contaminants, set to negative to omit')
            parser.add_option('--salt2betaprior',
                              default=map(float,config.get('all','salt2betaprior').split(',')),type='float',
                              help="""comma-separated gaussian prior on fraction of contaminants: centroid, sigma.
For flat prior, use empty string""",nargs=2)
            parser.add_option('--salt2betafixed', default=config.get('all','salt2betafixed'),type='int',
                              help="""0 to return posterior frac. of contaminants, 1 to keep fixed""")


            parser.add_option('--salt2alpha', default=config.get('all','salt2alpha'), type="float",
                              help='SALT2 alpha parameter from a spectroscopic sample (default=%default)')
            parser.add_option('--salt2alphaerr', default=config.get('all','salt2alphaerr'), type="float",
                              help='nominal SALT2 alpha uncertainty from a spectroscopic sample (default=%default)')
            parser.add_option('--salt2beta', default=config.get('all','salt2beta'), type="float",
                              help='nominal SALT2 beta parameter from a spec. sample (default=%default)')
            parser.add_option('--salt2betaerr', default=config.get('all','salt2betaerr'), type="float",
                              help='nominal SALT2 beta uncertainty from a spec. sample (default=%default)')
        
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

        else:
            parser.add_option('--clobber', default=False, action="store_true",
                              help='clobber output file')
            parser.add_option('--append', default=False, action="store_true",
                              help='open output file in append mode')

            # Input file
            parser.add_option('--pacol', default='PA', type="string",
                              help='column in input file used as prior P(A)')

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


            # fraction of contaminants: guesses and priors
            parser.add_option('--fracBguess', default=0.05,type='float',
                              help='initial guess for fraction of contaminants, set to negative to omit')
            parser.add_option('--fracBprior', default=(0.05,0.1),type='float',
                              help="""comma-separated gaussian prior on fraction of contaminants: centroid, sigma.  
For flat prior, use empty string""",nargs=2)
            parser.add_option('--fracBfixed', default=0,type='int',
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
            parser.add_option('--salt2betaprior', default=(3.13,1.0),type='float',
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
            parser.add_option('--nthreads', default=8, type="int",
                              help='Number of threads for MCMC')
            parser.add_option('--nwalkers', default=100, type="int",
                              help='Number of walkers for MCMC')
            parser.add_option('--nsteps', default=500, type="int",
                              help='Number of steps for MCMC')
            parser.add_option('--ninit', default=50, type="int",
                              help="Number of steps before the samples wander away from the initial values and are 'burnt in'")


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

        # open the output file
        if not os.path.exists(self.options.outputfile) or self.options.clobber:
            writeout = True
            fout = open(self.options.outputfile,'w')
            print >> fout, "# muA muAerr muAerr_m muAerr_p sigA sigAerr sigAerr_m sigAerr_p muB muBerr muBerr_m muBerr_p sigB sigBerr sigBerr_m sigBerr_p fracB fracBerr fracBerr_m fracBerr_p lstep lsteperr lsteperr_m lsteperr_p"""
            fout.close()
        elif not self.options.append:
            writeout = False
            print('Warning : files %s exists!!  Not clobbering'%self.options.outputfile)

        # run the MCMC
        residA,sigA,residB,sigB,fracB,lstep,samples = self.mcmc(inp)
                
        outlinefmt = "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f "
        outlinefmt += "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f"

        chain_len = len(samples[:,0])
        residAmean = np.mean(samples[:,0])
        residAerr = np.sqrt(np.sum((samples[:,0]-np.mean(samples[:,0]))*(samples[:,0]-np.mean(samples[:,0])))/chain_len)

        sigAmean = np.mean(samples[:,1])
        sigAerr = np.sqrt(np.sum((samples[:,1]-np.mean(samples[:,1]))*(samples[:,1]-np.mean(samples[:,1])))/chain_len)

        residBmean = np.mean(samples[:,2])
        residBerr = np.sqrt(np.sum((samples[:,2]-np.mean(samples[:,2]))*(samples[:,2]-np.mean(samples[:,2])))/chain_len)

        sigBmean = np.mean(samples[:,3])
        sigBerr = np.sqrt(np.sum((samples[:,3]-np.mean(samples[:,3]))*(samples[:,3]-np.mean(samples[:,3])))/chain_len)

        count = 4
        if self.options.fracBguess == 0:
            fracBmean = np.mean(samples[:,4])
            fracBerr = np.sqrt(np.sum((samples[:,4]-np.mean(samples[:,4]))*(samples[:,4]-np.mean(samples[:,4])))/chain_len)
            count += 1

        lstepmean = np.mean(samples[:,count])
        lsteperr = np.sqrt(np.sum((samples[:,count]-np.mean(samples[:,count]))*(samples[:,count]-np.mean(samples[:,count])))/chain_len)
        count += 1

        salt2alphamean = np.mean(samples[:,count])
        salt2alphaerr = np.sqrt(np.sum((samples[:,count]-np.mean(samples[:,count]))*(samples[:,count]-np.mean(samples[:,count])))/chain_len)
        count += 1

        salt2betamean = np.mean(samples[:,count])
        salt2betaerr = np.sqrt(np.sum((samples[:,count]-np.mean(samples[:,count]))*(samples[:,count]-np.mean(samples[:,count])))/chain_len)
        count += 1

        outline = outlinefmt%(residAmean,residAerr,residA[1],residA[2],
                              sigAmean,sigAerr,sigA[1],sigA[2],
                              residBmean,residBerr,residB[1],residB[2],
                              sigBmean,sigBerr,sigB[1],sigB[2],
                              fracBmean,fracBerr,fracB[1],fracB[2],
                              lstepmean,lsteperr,lstep[1],lstep[2],
                              alpha[0],alpha[1],alpha[2],
                              beta[0],beta[1],beta[2])
        if self.options.append or not os.path.exists(self.options.outputfile) or self.options.clobber:
            fout = open(self.options.outputfile,'a')
            print >> fout, outline
            fout.close()

        if self.options.verbose:
            print('muA: %.3f +/- %.3f muB: %.3f +/- %.3f frac. B: %.3f +/- %.3f Lstep: %.3f +/- %.3f salt2 alpha: %.3f +/- %.3f salt2 beta %.3f +/- %.3f'%(
                    residAmean,residAerr,
                    residBmean,residBerr,
                    fracBmean,fracBerr,
                    lstepmean,lsteperr,
                    salt2alphamean,salt2alphaerr,
                    salt2betamean,salt2betaerr))

    def mcmc(self,inp):
        from scipy.optimize import minimize
        import emcee

        if not inp.__dict__.has_key('PL'):
            inp.PL = 0

        # minimize, not maximize
        if self.options.fracBguess >= 0:
            omitfracB = False
        else:
            omitfracB = True
        nll = lambda *args: -twogausslike_nofrac(*args)
        inp.MU,inp.residerr = salt2mu(x1=inp.x1,x1err=inp.x1ERR,c=inp.c,cerr=inp.cERR,mb=inp.mB,mberr=inp.mBERR,
                                      cov_x1_c=inp.COV_x1_c,cov_x1_x0=inp.COV_x1_x0,cov_c_x0=inp.COV_c_x0,
                                      alpha=self.options.salt2alpha,alphaerr=self.options.salt2alphaerr,
                                      beta=self.options.salt2beta,betaerr=self.options.salt2betaerr,
                                      x0=inp.x0,sigint=self.options.sigint,z=inp.zHD)
        inp.resid = inp.MU - cosmo.mu(inp.zHD)
        md = minimize(nll,(self.options.popAguess[0],self.options.popAguess[1],
                           self.options.popBguess[0],self.options.popBguess[1],
                           self.options.fracBguess,self.options.lstepguess,
                           self.options.salt2alphaguess,self.options.salt2betaguess),
                      args=(inp.PA,inp.PL,inp.resid,inp.residerr))
        if md.message != 'Optimization terminated successfully.':
            print("""Warning : Minimization Failed!!!
Try some different initial guesses, or let the MCMC try and take care of it""")

        # for the fixed parameters, make really narrow priors
        md = self.fixedpriors(md)

        ndim, nwalkers = len(md["x"]), int(self.options.nwalkers)
        pos = [md["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                        args=(inp,self.options.salt2alpha,self.options.salt2alphaerr,
                                              self.options.salt2beta,self.options.salt2betaerr,omitfracB,
                                              self.options.p_residA,self.options.psig_residA,
                                              self.options.p_residB,self.options.psig_residB,
                                              self.options.p_sigA,self.options.psig_sigA,
                                              self.options.p_sigB,self.options.psig_sigB,
                                              self.options.p_fracB,self.options.psig_fracB,
                                              self.options.p_lstep,self.options.psig_lstep,
                                              self.options.p_salt2alpha,self.options.psig_salt2alpha,
                                              self.options.p_salt2beta,self.options.psig_salt2beta),
                                        threads=int(self.options.nthreads))
        pos, prob, state = sampler.run_mcmc(pos, 200)
        sampler.reset()
        sampler.run_mcmc(pos, self.options.nsteps, thin=1)
        samples = sampler.flatchain#[:, self.options.ninit:, :].reshape((-1, ndim))

        # get the error bars - should really return the full posterior!
        import scipy.stats
        resida_mcmc, siga_mcmc, residb_mcmc, sigb_mcmc, fracB, lstep, alpha, beta = \
            map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                zip(*np.percentile(samples, [16, 50, 84],
                                   axis=0)))
#        import pdb; pdb.set_trace()
        return(resida_mcmc,siga_mcmc,residb_mcmc,sigb_mcmc,fracB,lstep,alpha,beta,samples)

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
        if self.options.fracBfixed:
            self.options.p_fracB = self.options.fracBguess[0]
            self.options.psig_fracB = 1e-3
            md["x"][4] = self.options.fracBguess[0]
        if self.options.lstepfixed:
            self.options.p_lstep = self.options.lstepguess
            self.options.psig_lstep = 1e-3
            md["x"][5] = self.options.lstepguess

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
        self.options.p_lstep = self.options.lstepprior[0]
        self.options.psig_lstep = self.options.lstepprior[1]
        self.options.p_salt2alpha = self.options.salt2alphaprior[0]
        self.options.psig_salt2alpha = self.options.salt2alphaprior[1]
        self.options.p_salt2beta = self.options.salt2betaprior[0]
        self.options.psig_salt2beta = self.options.salt2betaprior[1]

def twogausslike_nofrac(x,PA=None,PL=None,resid=None,residerr=None):

    return np.sum(np.logaddexp(-(resid-x[0]+PL*x[5])**2./(2.0*(residerr**2.+x[1]**2.)) + \
                                    np.log(PA/(np.sqrt(2*np.pi)*np.sqrt(x[1]**2.+residerr**2.))),
                                -(resid-x[2])**2./(2.0*(residerr**2.+x[3]**2.)) + \
                                    np.log((1-PA)/(np.sqrt(2*np.pi)*np.sqrt(x[3]**2.+residerr**2.)))))

def twogausslike_snpars(x,inp=None,alpha=None,beta=None,alphaerr=None,betaerr=None):
    
    muA,residAerr = salt2mu(x1=inp.x1,x1err=inp.x1ERR,c=inp.c,cerr=inp.cERR,mb=inp.mB,mberr=inp.mBERR,
                         cov_x1_c=inp.COV_x1_c,cov_x1_x0=inp.COV_x1_x0,cov_c_x0=inp.COV_c_x0,
                         alpha=theta[-2],alphaerr=1e-10,
                         beta=theta[-1],betaerr=1e-10,
                         x0=inp.x0,sigint=self.options.sigint,z=inp.zHD)
    residA = muA - cosmo.mu(inp.zHD)

    muB,residBerr = salt2mu(x1=inp.x1,x1err=inp.x1ERR,c=inp.c,cerr=inp.cERR,mb=inp.mB,mberr=inp.mBERR,
                         cov_x1_c=inp.COV_x1_c,cov_x1_x0=inp.COV_x1_x0,cov_c_x0=inp.COV_c_x0,
                         alpha=alpha,alphaerr=alphaerr,
                         beta=beta,betaerr=betaerr,
                         x0=inp.x0,sigint=self.options.sigint,z=inp.zHD)
    residB = muB - cosmo.mu(inp.zHD)

    
    return np.sum(np.logaddexp(-(residA-x[0]+PL*x[5])**2./(2.0*(residAerr**2.+x[1]**2.)) + \
                                    np.log((1-x[4])*(PA)/(np.sqrt(2*np.pi)*np.sqrt(x[1]**2.+residAerr**2.))),
                                -(residB-x[2])**2./(2.0*(residBerr**2.+x[3]**2.)) + \
                                    np.log((x[4])*(1-PA)/(np.sqrt(2*np.pi)*np.sqrt(x[3]**2.+residBerr**2.)))))


def twogausslike_snpars_nofrac(x,inp=None,alpha=None,beta=None,alphaerr=None,betaerr=None):

    muA,residAerr = salt2mu(x1=inp.x1,x1err=inp.x1ERR,c=inp.c,cerr=inp.cERR,mb=inp.mB,mberr=inp.mBERR,
                         cov_x1_c=inp.COV_x1_c,cov_x1_x0=inp.COV_x1_x0,cov_c_x0=inp.COV_c_x0,
                         alpha=theta[-2],alphaerr=1e-10,
                         beta=theta[-1],betaerr=1e-10,
                         x0=inp.x0,sigint=self.options.sigint,z=inp.zHD)
    residA = muA - cosmo.mu(inp.zHD)

    muB,residBerr = salt2mu(x1=inp.x1,x1err=inp.x1ERR,c=inp.c,cerr=inp.cERR,mb=inp.mB,mberr=inp.mBERR,
                         cov_x1_c=inp.COV_x1_c,cov_x1_x0=inp.COV_x1_x0,cov_c_x0=inp.COV_c_x0,
                         alpha=alpha,alphaerr=alphaerr,
                         beta=beta,betaerr=betaerr,
                         x0=inp.x0,sigint=self.options.sigint,z=inp.zHD)
    residB = muB - cosmo.mu(inp.zHD)

    return np.sum(np.logaddexp(-(residA-x[0]+PL*x[5])**2./(2.0*(residAerr**2.+x[1]**2.)) + \
                                    np.log(PA/(np.sqrt(2*np.pi)*np.sqrt(x[1]**2.+residAerr**2.))),
                                -(residB-x[2])**2./(2.0*(residBerr**2.+x[3]**2.)) + \
                                    np.log((1-PA)/(np.sqrt(2*np.pi)*np.sqrt(x[3]**2.+residBerr**2.)))))

def lnprior(theta,
            p_residA=None,psig_residA=None,
            p_residB=None,psig_residB=None,
            p_sig_A=None,psig_sig_A=None,
            p_sig_B=None,psig_sig_B=None,
            p_fracB=None,psig_fracB=None,
            p_lstep=None,psig_lstep=None,
            p_salt2alpha=None,psig_salt2alpha=None,
            p_salt2beta=None,psig_salt2beta=None):

    try:
        residA,sigA,residB,sigB,fracB,lstep,alpha,beta = theta
    except:
        residA,sigA,residB,sigB,lstep,alpha,beta = theta

    p_theta = 1.0

    if p_residA:
        p_theta *= gauss(residA,p_residA,psig_residA)
    if p_residB:
        p_theta *= gauss(residB,p_residB,psig_residB)
    if p_sig_A:
        p_theta *= gauss(sigA,p_sig_A,psig_sig_A)
    if p_sig_B:
        p_theta *= gauss(sigB,p_sig_B,psig_sig_B)
    if p_fracB:
        p_theta *= gauss(fracB,p_fracB,psig_fracB)
    if p_salt2alpha:
        p_theta *= gauss(alpha,p_salt2alpha,psig_salt2beta)
    if p_salt2beta:
        p_theta *= gauss(beta,p_salt2alpha,psig_salt2beta)
    if type(p_lstep) != None:
        p_lstep *= gauss(lstep,p_lstep,psig_lstep)


    if fracB > 1 or fracB < 0: return -np.inf
    else: return(np.log(p_theta))

def lnprob(theta,inp=None,
           alpha=None,alphaerr=None,
           beta=None,betaerr=None,
           omitfracB=False,
           p_residA=None,psig_residA=None,
           p_residB=None,psig_residB=None,
           p_sig_A=None,psig_sig_A=None,
           p_sig_B=None,psig_sig_B=None,
           p_fracB=None,psig_fracB=None,
           p_lstep=None,psig_lstep=None,
           p_salt2alpha=None,psig_salt2alpha=None,
           p_salt2beta=None,psig_salt2beta=None):

    lp = lnprior(theta,
                 p_residA=p_residA,psig_residA=psig_residA,
                 p_residB=p_residB,psig_residB=psig_residB,
                 p_sig_A=p_sig_A,psig_sig_A=psig_sig_A,
                 p_sig_B=p_sig_B,psig_sig_B=psig_sig_B,
                 p_fracB=p_fracB,psig_fracB=psig_fracB,
                 p_lstep=p_lstep,psig_lstep=psig_lstep,
                 p_salt2alpha=p_salt2alpha,psig_salt2alpha=psig_salt2alpha,
                 p_salt2beta=p_salt2beta,psig_salt2beta=psig_salt2beta)
    if not np.isfinite(lp) or np.isnan(lp):
        return -np.inf

    if not omitfracB:
        post = lp+twogausslike_snpars(theta,inp,alpha,alphaerr,beta,betaerr)
    else:
        post = lp+twogausslike_snpars_nofrac(
            theta,inp,alpha,alphaerr,beta,betaerr)

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
    usagestring="""An implementation of the BEAMS method that constrains the SN 
parameters alpha and beta (Kunz, Bassett, & Hlozek 2006).  Uses Bayesian methods 
to estimate the mags of a sample with contaminants.  Takes a
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
    import cosmo
    from SNCosmo import salt2mu
    
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
