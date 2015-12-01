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
        parser.add_option('--clobber', default=False, action="store_true",
                          help='clobber output file')
        parser.add_option('--append', default=False, action="store_true",
                          help='open output file in append mode')

        if config:
            # Input file
            parser.add_option('--pacol', default=config.get('all','pacol'), type="string",
                              help='column in input file used as prior P(A)')
            parser.add_option('--mucol', default=config.get('all','mucol'), type="string",
                              help='column name in input file header for distance modulus')
            parser.add_option('--muerrcol', default=config.get('all','muerrcol'), type="string",
                              help='column name in input file header for distance modulus errors')
            parser.add_option('--zcol', default=config.get('all','zcol'), type="string",
                              help='column name in input file header for z')


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

        
            # output and number of threads
            parser.add_option('--nthreads', default=config.get('all','nthreads'), type="int",
                              help='Number of threads for MCMC')
            parser.add_option('--nwalkers', default=config.get('all','nwalkers'), type="int",
                              help='Number of walkers for MCMC')
            parser.add_option('--nsteps', default=config.get('all','nsteps'), type="int",
                              help='Number of steps for MCMC')
            parser.add_option('--ninit', default=config.get('all','ninit'), type="int",
                              help="Number of steps before the samples wander away from the initial values and are 'burnt in'")


            parser.add_option('--nzbins', default=config.get('all','nzbins'), type="int",
                              help='Number of z bins')
            parser.add_option('--nzskip', default=config.get('all','nzbins'), type="int",
                              help='This is a hack - don\t write the first x bins to the output files')
            parser.add_option('--zmin', default=config.get('all','zmin'), type="float",
                              help='min redshift')
            parser.add_option('--zmax', default=config.get('all','zmax'), type="float",
                              help='max redshift')
            parser.add_option('--equalbins', default=config.get('all','equalbins'), action="store_true",
                              help='if set, divide bins so that there are the same number of SNe in each bin')


            parser.add_option('-i','--inputfile', default=config.get('all','inputfile'), type="string",
                              help='file with the input data')
            parser.add_option('-o','--outputfile', default=config.get('all','outputfile'), type="string",
                              help='Output file with the derived parameters for each redshift bin')
            parser.add_option('--covmatfile', default=config.get('all','covmatfile'), type="string",
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
            parser.add_option('--popAprior_mean', default=(0.0,0.2),type='float',
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
            
        
            # output and number of threads
            parser.add_option('--nthreads', default=8, type="int",
                              help='Number of threads for MCMC')
            parser.add_option('--nzskip', default=0, type="int",
                              help='This is a hack - don\t write the first x bins to the output files')
            parser.add_option('--nwalkers', default=200, type="int",
                              help='Number of walkers for MCMC')
            parser.add_option('--nsteps', default=500, type="int",
                              help='Number of steps for MCMC')
            parser.add_option('--ninit', default=50, type="int",
                              help="Number of steps before the samples wander away from the initial values and are 'burnt in'")


            parser.add_option('--nzbins', default=30, type="int",
                              help='Number of z bins')
            parser.add_option('--zmin', default=0.02, type="float",
                              help='min redshift')
            parser.add_option('--zmax', default=0.7, type="float",
                              help='max redshift')
            parser.add_option('--equalbins', default=False, action="store_true",
                              help='if set, divide bins so that there are the same number of SNe in each bin')

            parser.add_option('-i','--inputfile', default='BEAMS.input', type="string",
                              help='file with the input data')
            parser.add_option('-o','--outputfile', default='beamsCosmo.out', type="string",
                              help='Output file with the derived parameters for each redshift bin')
            parser.add_option('--covmatfile', default='BEAMS.covmat', type="string",
                              help='Output covariance matrix')


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
        inp.mu = inp.__dict__[self.options.mucol]
        inp.muerr = inp.__dict__[self.options.muerrcol]

        # open the output file
        if not os.path.exists(self.options.outputfile) or self.options.clobber:
            writeout = True
            fout = open(self.options.outputfile,'w')
            print >> fout, "# muA muAerr sigA sigAerr_m sigAerr_p muB muBerr_m muBerr_p sigB sigBerr_m sigBerr_p fracB fracBerr_m fracBerr_p lstep lstep_m lstep_p"""
            fout.close()
        elif not self.options.append:
            writeout = False
            print('Warning : files %s exists!!  Not clobbering'%self.options.outputfile)

        # run the MCMC
        zcontrol = np.logspace(np.log10(self.options.zmin),np.log10(self.options.zmax),num=self.options.nzbins)
        if self.options.equalbins:
            from scipy import stats
            zcontrol = stats.mstats.mquantiles(inp.z,np.arange(0,1,1./self.options.nzbins))

        cov,params = self.mcmc(inp,zcontrol)
        if self.options.covmatfile:
            fout = open(self.options.covmatfile,'w')
            print >> fout, '%i'%len(cov)
            shape = np.shape(cov)[0]
            for j in range(shape):
                outline = ''
                for i in range(shape):
                    outline += '%8.5e '%cov[j,i]
                    print >> fout, '%8.5e'%cov[j,i]#outline
            fout.close()

        # residA,sigA,residB,sigB,fracB,lstep = self.mcmc(inp)
        sigA,residB,sigB,fracB,lstep = params[:5]
        mupar = params[5:]

        outlinefmt = "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f"
        for par,zcntrl,i in zip(mupar[self.options.nzskip:],zcontrol[self.options.nzskip:],range(len(mupar[self.options.nzskip:]))):
            outline = outlinefmt%(par[0],np.sqrt(cov[i,i]),
                                  sigA[0],sigA[1],sigA[2],
                                  residB[0],residB[1],residB[2],
                                  sigB[0],sigB[1],sigB[2],
                                  fracB[0],fracB[1],fracB[2],
                                  lstep[0],lstep[1],lstep[2])
            if self.options.append or not os.path.exists(self.options.outputfile) or self.options.clobber:
                fout = open(self.options.outputfile,'a')
                print >> fout, outline
                fout.close()

            if self.options.verbose:
                print('z: %.3f muA: %.3f +/- %.3f muB: %.3f +/- %.3f frac. B: %.3f +/- %.3f Lstep: %.3f +/- %.3f'%(
                        zcntrl,par[0],np.mean([par[1],par[2]]),
                        residB[0],np.mean([residB[1],residB[2]]),
                        fracB[0],np.mean([fracB[1],fracB[2]]),
                        lstep[0],np.mean([lstep[1],lstep[2]])))


    def mcmc(self,inp,zcontrol):
        from scipy.optimize import minimize
        import emcee
        import cosmo
        if not inp.__dict__.has_key('PL'):
            inp.PL = 0

        # minimize, not maximize
        if self.options.fracBguess >= 0:
            omitfracB = False
        else:
            omitfracB = True
        nll = lambda *args: -twogausslike_nofrac(*args)

        guess = (self.options.popAguess[1],
                 self.options.popBguess[0],self.options.popBguess[1],
                 self.options.fracBguess,self.options.lstepguess,)
        for z in zcontrol:
            guess += (self.options.popAguess[0]+cosmo.mu(z),)
        md = minimize(nll,guess,
                      args=(inp.PA,inp.PL,inp.mu,inp.muerr,inp.z,zcontrol))
        if md.message != 'Optimization terminated successfully.':
            print("""Warning : Minimization Failed!!!  
Try some different initial guesses, or let the MCMC try and take care of it""")

        # for the fixed parameters, make really narrow priors
        md = self.fixedpriors(md)
        for i in range(len(md["x"])):
            if i >= 5: md["x"][i] = cosmo.mu(zcontrol[i-5])

        ndim, nwalkers = len(md["x"]), int(self.options.nwalkers)
        pos = [md["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
                                        args=(inp.PA,inp.PL,inp.mu,inp.muerr,inp.z,zcontrol,
                                              omitfracB,
                                              self.options.p_residA,self.options.psig_residA,
                                              self.options.p_residB,self.options.psig_residB,
                                              self.options.p_sigA,self.options.psig_sigA,
                                              self.options.p_sigB,self.options.psig_sigB,
                                              self.options.p_fracB,self.options.psig_fracB,
                                              self.options.p_lstep,self.options.psig_lstep),
                                        threads=int(self.options.nthreads))
        sampler.run_mcmc(pos, self.options.nsteps)
        samples = sampler.chain[:, self.options.ninit:, :].reshape((-1, ndim))

        # get the error bars - should really return the full posterior

        params = \
            map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                zip(*np.percentile(samples, [16, 50, 84],
                                   axis=0)))

        cov = covmat(samples[:,np.shape(samples)[1] - self.options.nzbins + self.options.nzskip:])
        return(cov,params)

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

def twogausslike(x,PA=None,PL=None,mu=None,muerr=None,z=None,zcontrol=None):

    mumodel = np.interp(np.log10(z),np.log10(zcontrol),x[5:])

#    mumodel = np.zeros(len(z))
#    for mub,mub1,zb,zb1 in zip(x[5:-1],x[6:],zcontrol[:-1],zcontrol[1:]):
#        cols = np.where((z >= zb) & (z < zb1))[0]
#        alpha = np.log10(z[cols]/zb)/np.log10(zb1/zb)
#        mumodel[cols] = (1-alpha)*mub + alpha*mub1

    return np.sum(np.logaddexp(-(mu-mumodel+PL*x[4])**2./(2.0*(muerr**2.+x[0]**2.)) + \
                                    np.log((1-x[3])*(PA)/(np.sqrt(2*np.pi)*np.sqrt(x[0]**2.+muerr**2.))),
                                -(mu-mumodel-x[1])**2./(2.0*(muerr**2.+x[2]**2.)) + \
                                    np.log((x[3])*(1-PA)/(np.sqrt(2*np.pi)*np.sqrt(x[2]**2.+muerr**2.)))))


def twogausslike_nofrac(x,PA=None,PL=None,mu=None,muerr=None,z=None,zcontrol=None):

    mumodel = np.interp(np.log10(z),np.log10(zcontrol),x[5:])

#    mumodel = np.zeros(len(z))
#    for mub,mub1,zb,zb1 in zip(x[5:-1],x[6:],zcontrol[:-1],zcontrol[1:]):
#        cols = np.where((z >= zb) & (z < zb1))[0]
#        alpha = np.log10(z[cols]/zb)/np.log10(zb1/zb)
#        mumodel[cols] = (1-alpha)*mub + alpha*mub1

    return np.sum(np.logaddexp(-(mu-mumodel+PL*x[3])**2./(2.0*(muerr**2.+x[0]**2.)) + \
                                    np.log(PA/(np.sqrt(2*np.pi)*np.sqrt(x[0]**2.+muerr**2.))),0))
#                                -(mu-mumodel-x[1])**2./(2.0*(muerr**2.+x[2]**2.)) + \
#                                    np.log((1-PA)/(np.sqrt(2*np.pi)*np.sqrt(x[2]**2.+muerr**2.)))))

def lnprior(theta,
            p_residA=None,psig_residA=None,
            p_residB=None,psig_residB=None,
            p_sig_A=None,psig_sig_A=None,
            p_sig_B=None,psig_sig_B=None,
            p_fracB=None,psig_fracB=None,
            p_lstep=None,psig_lstep=None,
            omitfracB=False,zcntrl=None):
    import cosmo

    if not omitfracB:
        sigA,residB,sigB,fracB,lstep = theta[:5]
        residAlist = theta[5:]
    else:
        sigA,residB,sigB,lstep = theta[:4]
        residAlist = theta[4:]

    p_theta = 1.0

    if p_residA or p_residA == 0.0:
        for residA,z in zip(residAlist,zcntrl):
            #print 'hi'
            #print p_theta,residA,p_residA,cosmo.mu(z),psig_residA
            p_theta *= gauss(residA,p_residA+cosmo.mu(z),psig_residA)
            #print residA,p_residA+cosmo.mu(z),p_theta,gauss(residA,p_residA+cosmo.mu(z),psig_residA)
        #print 'done',p_theta
    if p_residB:
        p_theta *= gauss(residB,p_residB,psig_residB)
    if p_sig_A:
        p_theta *= gauss(sigA,p_sig_A,psig_sig_A)
    if p_sig_B:
        p_theta *= gauss(sigB,p_sig_B,psig_sig_B)
    if p_fracB:
        p_theta *= gauss(fracB,p_fracB,psig_fracB)
    if type(p_lstep) != None:
        p_lstep *= gauss(lstep,p_lstep,psig_lstep)


    if fracB > 1 or fracB < 0: return -np.inf
    else: return(np.log(p_theta))

def lnprob(theta,PA=None,PL=None,mu=None,muerr=None,
           z=None,zcontrol=None,
           omitfracB=False,
           p_residA=None,psig_residA=None,
           p_residB=None,psig_residB=None,
           p_sig_A=None,psig_sig_A=None,
           p_sig_B=None,psig_sig_B=None,
           p_fracB=None,psig_fracB=None,
           p_lstep=None,psig_lstep=None):

    lp = lnprior(theta,
                 p_residA=p_residA,psig_residA=psig_residA,
                 p_residB=p_residB,psig_residB=psig_residB,
                 p_sig_A=p_sig_A,psig_sig_A=psig_sig_A,
                 p_sig_B=p_sig_B,psig_sig_B=psig_sig_B,
                 p_fracB=p_fracB,psig_fracB=psig_fracB,
                 p_lstep=p_lstep,psig_lstep=psig_lstep,
                 omitfracB=omitfracB,zcntrl=zcontrol)

    if not np.isfinite(lp) or np.isnan(lp):
        return -np.inf

    if omitfracB:
        post = lp+twogausslike(theta,PA=PA,PL=PL,
                               mu=mu,muerr=muerr,
                               z=z,zcontrol=zcontrol)
    else:
        post = lp+twogausslike_nofrac(
            theta,PA=PA,PL=PL,mu=mu,muerr=muerr,
            z=z,zcontrol=zcontrol)

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

