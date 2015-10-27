#!/usr/bin/env python
# D. Jones - 9/1/15
"""BEAMS method for PS1 data"""
import numpy as np

class sncosmo:
    def __init__(self):
        self.clobber = False
        self.verbose = False

    def add_options(self, parser=None, usage=None):
        import optparse
        if parser == None:
            parser = optparse.OptionParser(usage=usage, conflict_handler="resolve")

        parser.add_option('-v', '--verbose', action="count", dest="verbose",default=1)
        parser.add_option('--debug', default=False, action="store_true",
                          help='debug mode: more output and debug files')
        parser.add_option('--clobber', default=False, action="store_true",
                          help='clobber output image')

        parser.add_option('--lowzOm', default=0.3, type="float",
                          help='Omega matter used to determine slope across each redshift bin')
        parser.add_option('--lowzOde', default=0.7, type="float",
                          help='Omega Lambda used to determine slope across each redshift bin')
        parser.add_option('--lowzw', default=-1.0, type="float",
                          help='w used to determine slope across each redshift bin')

        parser.add_option('--piacol', default='FITPROB', type="string",
                          help='Column in FITRES file used as guess at P(Ia)')
        parser.add_option('--specconfcol', default=None, type="string",
                          help='Column in FITRES file indicating spec.-confirmed SNe with 1')

        # Light curve cut parameters
        parser.add_option(
            '--crange', default=(-0.3,0.3),type="float",
            help='Peculiar velocity error (default=%default)',nargs=2)
        parser.add_option(
            '--x1range', default=(-3.0,3.0),type="float",
            help='Peculiar velocity error (default=%default)',nargs=2)
        parser.add_option('--x1ccircle',default=False,action="store_true",
            help='Circle cut in x1 and c')
        parser.add_option(
            '--zrange', default=(0.023,1.0),type="float",
            help='allowed redshift range (default=%default)',nargs=2)
        parser.add_option(
            '--fitprobmin', default=0.001,type="float",
            help='Peculiar velocity error (default=%default)')
        parser.add_option(
            '--x1errmax', default=1.0,type="float",
            help='Peculiar velocity error (default=%default)')
        parser.add_option(
            '--pkmjderrmax', default=2.0,type="float",
            help='Peculiar velocity error (default=%default)')

        # SALT2 parameters and intrinsic dispersion
        parser.add_option('--salt2alpha', default=0.147, type="float",
                          help='SALT2 alpha parameter from a spectroscopic sample')
        parser.add_option('--salt2alphaerr', default=0.010, type="float",
                          help='nominal SALT2 alpha uncertainty from a spectroscopic sample')
        parser.add_option('--salt2beta', default=3.13, type="float",
                          help='nominal SALT2 beta parameter from a spec. sample')
        parser.add_option('--salt2betaerr', default=0.12, type="float",
                          help='nominal SALT2 beta uncertainty from a spec. sample')
        parser.add_option('--fitsalt2pars', default=False, action="store_true",
                          help='If set, determine SALT2 nuisance parameters with MCMC.  Otherwise, use values derived from spec. sample')
        parser.add_option('--sigint', default=0.0, type="float",
                          help='nominal intrinsic dispersion, MCMC fits for this if not specified')

        # Mass options
        parser.add_option(
            '--masscorr', default=False,type="int",
            help='If true, perform mass correction (default=%default)')
        parser.add_option(
            '--massfile', default='mass.txt',
            type="string",help='Host mass file (default=%default)')
        parser.add_option(
            '--masscorrmag', default=(0.06,0.023),type="float",
            help="""mass corr. and uncertainty (default=%default)""",nargs=2)

        parser.add_option('--nthreads', default=8, type="int",
                          help='Number of threads for MCMC')
        parser.add_option('--zmin', default=0.0, type="float",
                          help='minimum redshift')
        parser.add_option('--zmax', default=0.7, type="float",
                          help='maximum redshift')
        parser.add_option('--zbinsize', default=0.05, type="float",
                          help='bin size in redshift space')


        parser.add_option('-f','--fitresfile', default='ps1_psnidprob.fitres', type="string",
                          help='fitres file with the SN Ia data')
        parser.add_option('-o','--outfile', default='beamsCosmo.out', type="string",
                          help='Output file with the derived parameters for each redshift bin')

        parser.add_option('--mkplot', default=False, action="store_true",
                          help='plot the results')
        parser.add_option('--usefitresmu', default=False, action="store_true",
                          help='User the distance modulus given in the fitres file')

        parser.add_option('-p','--paramfile', default='', type="string",
                          help='fitres file with the SN Ia data')

        return(parser)

    def main(self,fitres):
        from txtobj import txtobj
        import cosmo

        fr = txtobj(fitres,fitresheader=True)
        fr.MU,fr.MUERR = salt2mu(x1=fr.x1,x1err=fr.x1ERR,c=fr.c,cerr=fr.cERR,mb=fr.mB,mberr=fr.mBERR,
                                 cov_x1_c=fr.COV_x1_c,cov_x1_x0=fr.COV_x1_x0,cov_c_x0=fr.COV_c_x0,
                                 alpha=self.options.salt2alpha,alphaerr=self.options.salt2alphaerr,
                                 beta=self.options.salt2beta,betaerr=self.options.salt2betaerr,
                                 x0=fr.x0)

        # Light curve cuts
        if self.options.x1ccircle:
            # I'm just going to assume cmax = abs(cmin) and same for x1
            cols = np.where((fr.x1**2./self.options.x1range[0]**2. + fr.c**2./self.options.crange[0]**2. < 1) &
                            (fr.x1ERR < self.options.x1errmax) & (fr.PKMJDERR < self.options.pkmjderrmax/(1+fr.zHD)) &
                            (fr.FITPROB > self.options.fitprobmin) &
                            (fr.z > self.options.zrange[0]) & (fr.z < self.options.zrange[1]))
        else:
            cols = np.where((fr.x1 > self.options.x1range[0]) & (fr.x1 < self.options.x1range[1]) &
                            (fr.c > self.options.crange[0]) & (fr.c < self.options.crange[1]) &
                            (fr.x1ERR < self.options.x1errmax) & (fr.PKMJDERR < self.options.pkmjderrmax) &
                            (fr.FITPROB > self.options.fitprobmin) &
                            (fr.z > self.options.zrange[0]) & (fr.z < self.options.zrange[1]))
        for k in fr.__dict__.keys():
            fr.__dict__[k] = fr.__dict__[k][cols]

        # Prior SN Ia probabilities
        P_Ia = np.zeros(len(fr.CID))
        for i in range(len(fr.CID)):
            P_Ia[i] = fr.__dict__[self.options.piacol][i]
            if self.options.specconfcol:
                if fr.__dict__[self.options.specconfcol][i] == 1:
                    P_Ia[i] = 1

        from doBEAMS import BEAMS
        import ConfigParser, sys
        sys.argv = ['./doBEAMS.py']
        beam = BEAMS()
        parser = beam.add_options()
        options,  args = parser.parse_args(args=None,values=None)
        options.paramfile = self.options.paramfile

        if options.paramfile:
            config = ConfigParser.ConfigParser()
            config.read(options.paramfile)
        else: config=None
        parser = beam.add_options(config=config)
        options,  args = parser.parse_args()

        beam.options = options
        beam.transformOptions()
        options.inputfile = 'BEAMS_SN.input'
        if self.options.masscorr: beam.options.lstep = True

        # loop over the redshift bins
        append = False
        for zmin,zmax in zip(np.arange(self.options.zmin,self.options.zmax,self.options.zbinsize),
                             np.arange(self.options.zmin+self.options.zbinsize,self.options.zmax+self.options.zbinsize,self.options.zbinsize)):
            if zmin > self.options.zmin:
                append = True; clobber = False
            else: clobber = True

            # make the BEAMS input file
            fout = open('BEAMS_SN.input','w')
            print >> fout, '# PA resid resid_err'
            for i in range(len(fr.MU)):
                if fr.z[i] > zmin and fr.z[i] <= zmax:
                    print >> fout, '%.3f %.4f %.4f'%(P_Ia[i],fr.MU[i]-cosmo.mu(fr.z[i]),fr.MUERR[i])
            fout.close()

            beam.options.append = append
            beam.options.clobber = clobber
            beam.main(options.inputfile)

    def mkplot(self,fitresfile=None):
        
        import pylab as plt
        from txtobj import txtobj
        bms = txtobj(self.options.outfile)
        fr = txtobj(fitresfile,fitresheader=True)

        # Light curve cuts
        if self.options.x1ccircle:
            # I'm just going to assume cmax = abs(cmin) and same for x1
            cols = np.where((fr.x1**2./self.options.x1range[0]**2. + fr.c**2./self.options.crange[0]**2. < 1) &
                            (fr.x1ERR < self.options.x1errmax) & (fr.PKMJDERR < self.options.pkmjderrmax/(1+fr.zHD)) &
                            (fr.FITPROB > self.options.fitprobmin) &
                            (fr.z > self.options.zrange[0]) & (fr.z < self.options.zrange[1]))
        else:
            cols = np.where((fr.x1 > self.options.x1range[0]) & (fr.x1 < self.options.x1range[1]) &
                            (fr.c > self.options.crange[0]) & (fr.c < self.options.crange[1]) &
                            (fr.x1ERR < self.options.x1errmax) & (fr.PKMJDERR < self.options.pkmjderrmax) &
                            (fr.FITPROB > self.options.fitprobmin) &
                            (fr.z > self.options.zrange[0]) & (fr.z < self.options.zrange[1]))
        for k in fr.__dict__.keys():
            fr.__dict__[k] = fr.__dict__[k][cols]

        plt.ion()
        plt.rcParams['figure.figsize'] = (16,8)
        plt.clf()
        ax2 = plt.axes([0.08,0.08,0.65,0.3])
        plt.axhline(y=0.0,color='0.2')
        ax1 = plt.axes([0.08,0.45,0.85,0.5])
        ax3 = plt.axes([0.73,0.08,0.2,0.3],sharey=ax2)
        plt.axhline(y=0.0,color='0.2')

        # loop over the redshift bins
        import cosmo
        zcosmo = np.arange(0.0,1.0,0.001)
        ax1.errorbar(zcosmo,cosmo.mu(zcosmo),color='k')
        mu_cc_p50,mu_ia_p50,z_ia_p50,z_cc_p50 = np.array([]),np.array([]),np.array([]),np.array([])
        for zmin,zmax,i in zip(np.arange(0,0.7,0.05),np.arange(0.05,0.75,0.05),range(len(np.arange(0.05,0.75,0.05)))):
            cols = np.where((fr.z > zmin) & (fr.z < zmax))
            zbin = np.mean([zmin,zmax])

            if not self.options.usefitresmu:
                mu,muerr = salt2mu(x1=fr.x1[cols],x1err=fr.x1ERR[cols],
                                   c=fr.c[cols],cerr=fr.cERR[cols],
                                   mb=fr.mB[cols],mberr=fr.mBERR[cols],
                                   alpha=bms.alpha_Ia[i],beta=bms.beta_Ia[i],
                                   alphaerr=np.mean([bms.alphaerr_Ia_m[i],bms.alphaerr_Ia_p[i]]),
                                   betaerr=np.mean([bms.betaerr_Ia_m[i],bms.betaerr_Ia_p[i]]))
            else: mu,muerr = fr.MU[cols],fr.MUERR[cols]

            P_Ia_post = gauss(mu,bms.mu_Ia[i],np.sqrt(muerr**2.+bms.sig_Ia[i]**2.))
            P_CC_post = gauss(mu,bms.mu_nonIa[i],np.sqrt(muerr**2.+bms.sig_nonIa[i]**2.))
            P_Ia_final = P_Ia_post/(P_Ia_post+P_CC_post)
            mu_cc_p50 = np.append(mu_cc_p50,mu[np.where(P_Ia_final < 0.5)])
            mu_ia_p50 = np.append(mu_ia_p50,mu[np.where(P_Ia_final > 0.5)])
            z_cc_p50 = np.append(z_cc_p50,fr.z[cols][np.where(P_Ia_final < 0.5)])
            z_ia_p50 = np.append(z_ia_p50,fr.z[cols][np.where(P_Ia_final > 0.5)])


            ax1.errorbar(fr.z[cols],mu,yerr=muerr,fmt='o',color='0.5',alpha=0.5)
            sc = ax1.scatter(fr.z[cols],mu,c=P_Ia_final,s=30,zorder=9,cmap='RdBu_r',vmin=0.0,vmax=1.0,alpha=0.6)
            ax2.errorbar(fr.z[cols],mu-cosmo.mu(fr.z[cols]),yerr=muerr,fmt='o',color='0.5',alpha=0.5)
            sc = ax2.scatter(fr.z[cols],mu-cosmo.mu(fr.z[cols]),c=P_Ia_final,s=30,zorder=9,cmap='RdBu_r',vmin=0.0,vmax=1.0,alpha=0.6)

            ax1.errorbar(bms.z[i],bms.mu_Ia[i],yerr=[[bms.muerr_Ia_m[i]],[bms.muerr_Ia_p[i]]],fmt='o',color='lightgreen',zorder=40,lw=3,ms=5,label='$\mu_{Ia}$')
            ax1.errorbar(bms.z[i],bms.mu_nonIa[i],yerr=[[bms.muerr_nonIa_m[i]],[bms.muerr_nonIa_p[i]]],fmt='o',color='b',zorder=40,lw=3,ms=5,label='$\mu_{CC}$')
            ax2.errorbar(bms.z[i],bms.mu_Ia[i]-cosmo.mu(zbin),yerr=[[bms.muerr_Ia_m[i]],[bms.muerr_Ia_p[i]]],fmt='o',color='lightgreen',zorder=40,lw=3,ms=5)
            ax2.errorbar(bms.z[i],bms.mu_nonIa[i]-cosmo.mu(zbin),yerr=[[bms.muerr_nonIa_m[i]],[bms.muerr_nonIa_p[i]]],fmt='o',color='b',zorder=40,lw=3,ms=5)
            if i == 0:
                ax1.legend(numpoints=1,loc='upper left')


        ax1.set_xlim([0.02,0.79])
        ax2.set_xlim([0.02,0.79])
        ax1.set_ylim([35,45])
        ax2.set_ylim([-3,3])

        cax = plt.axes([0.55,0.525,0.35,0.06])
        cb = plt.colorbar(sc,cax=cax,orientation='horizontal',ticks=[0,0.2,0.4,0.6,0.8,1.0])
        cax.set_xlabel('$P(Ia)$',labelpad=-65,fontsize='large')

        n_mu = np.histogram(mu_cc_p50-cosmo.mu(z_cc_p50),bins=30,range=[-3,3])
        ax3.plot(n_mu[0]/float(len(fr.MU)),n_mu[1][:-1],color='b',drawstyle='steps',lw=2)
        n_mu = np.histogram(mu_ia_p50-cosmo.mu(z_ia_p50),bins=30,range=[-3,3])
        ax3.plot(n_mu[0]/float(len(fr.MU)),n_mu[1][:-1],color='lightgreen',drawstyle='steps',lw=2)

        ax1.set_xlabel('$z$',fontsize=20)
        ax1.set_ylabel('$\mu$',fontsize=20)
        ax2.set_xlabel('$z$',fontsize=20)
        ax2.set_ylabel('$\mu - \mu_{\Lambda CDM}$',fontsize=20)
        ax3.xaxis.set_ticks([])

        import pdb; pdb.set_trace()

        
def gausshist(x,sigma=1,peak=1.,center=0):

    y = peak*np.exp(-(x-center)**2./(2.*sigma**2.))

    return(y)

def salt2mu(x1=None,x1err=None,
            c=None,cerr=None,
            mb=None,mberr=None,
            cov_x1_c=None,cov_x1_x0=None,cov_c_x0=None,
            alpha=None,beta=None,
            alphaerr=None,betaerr=None,
            M=None,x0=None):
    from uncertainties import ufloat, correlated_values, correlated_values_norm
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
        mu_out,muerr_out = np.append(mu_out,mu.n),np.append(muerr_out,mu.std_dev)

    return(mu_out,muerr_out)


if __name__ == "__main__":
    usagestring="""BEAMS method (Kunz et al. 2006) for PS1 data.
Uses Bayesian methods to estimate the true distance moduli of SNe Ia and
a second "other" species.  In this approach, I'll estimate this quantity in
rolling redshift bins at the location of each SN, using a nominal linear
fit at z > 0.1 and a cosmological fit to low-z spec data at z < 0.1.

Additional options are provided to doBEAMS.py with the parameter file.

USAGE: SNCosmo.py [options]

examples:
"""

    import exceptions
    import os
    import optparse

    sne = sncosmo()

    parser = sne.add_options(usage=usagestring)
    options,  args = parser.parse_args()

    sne.options = options
    sne.verbose = options.verbose
    sne.clobber = options.clobber

    from scipy.optimize import minimize
    import emcee
    import cosmo

    if not options.mkplot:
        sne.main(options.fitresfile)
    else:
        sne.mkplot(fitresfile=options.fitresfile)
