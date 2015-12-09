#!/usr/bin/env python
# D. Jones - 9/1/15
"""BEAMS method for PS1 data"""
import numpy as np

fitresheader = """# VERSION: PS1_PS1MD
# FITOPT:  NONE
# ---------------------------------------- 
NVAR: 30 
VARNAMES:  CID IDSURVEY TYPE FIELD zHD zHDERR z zERR HOST_LOGMASS HOST_LOGMASS_ERR SNRMAX1 SNRMAX2 SNRMAX3 PKMJD PKMJDERR x1 x1ERR c cERR mB mBERR x0 x0ERR COV_x1_c COV_x1_x0 COV_c_x0 NDOF FITCHI2 FITPROB 
# VERSION_SNANA      = v10_39i 
# VERSION_PHOTOMETRY = PS1_PS1MD 
# TABLE NAME: FITRES 
# 
"""
fitresvars = ["CID","IDSURVEY","TYPE","FIELD",
              "zHD","zHDERR","z","zERR","HOST_LOGMASS",
              "HOST_LOGMASS_ERR","SNRMAX1","SNRMAX2",
              "SNRMAX3","PKMJD","PKMJDERR","x1","x1ERR",
              "c","cERR","mB","mBERR","x0","x0ERR","COV_x1_c",
              "COV_x1_x0","COV_c_x0","NDOF","FITCHI2","FITPROB"]
fitresfmt = 'SN: %s %i %i %s %.5f %.5f %.5f %.5f %i %i %.4f %.4f %.4f %.3f %.3f %8.5e %8.5e %8.5e %8.5e %.4f %.4f %8.5e %8.5e %8.5e %8.5e %8.5e %i %.4f %.4f'

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
            '--fitprobmin', default=0.001,type="float",
            help='Peculiar velocity error (default=%default)')
        parser.add_option(
            '--x1errmax', default=1.0,type="float",
            help='Peculiar velocity error (default=%default)')
        parser.add_option(
            '--pkmjderrmax', default=2.0,type="float",
            help='Peculiar velocity error (default=%default)')

        # SALT2 parameters and intrinsic dispersion
        parser.add_option('--salt2alpha', default=0.147, type="float",#0.147
                          help='SALT2 alpha parameter from a spectroscopic sample (default=%default)')
        parser.add_option('--salt2alphaerr', default=0.01, type="float",#0.01
                          help='nominal SALT2 alpha uncertainty from a spectroscopic sample (default=%default)')
        parser.add_option('--salt2beta', default=3.13, type="float",#3.13
                          help='nominal SALT2 beta parameter from a spec. sample (default=%default)')
        parser.add_option('--salt2betaerr', default=0.12, type="float",#0.12
                          help='nominal SALT2 beta uncertainty from a spec. sample (default=%default)')
        parser.add_option('--fitsalt2pars', default=False, action="store_true",
                          help='If set, determine SALT2 nuisance parameters with MCMC.  Otherwise, use values derived from spec. sample')
        parser.add_option('--sigint', default=None, type="float",
                          help='nominal intrinsic dispersion, MCMC fits for this if not specified (default=%default)')

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
        parser.add_option('--zmin', default=0.01, type="float",
                          help='minimum redshift')
        parser.add_option('--zmax', default=0.75, type="float",
                          help='maximum redshift')
        parser.add_option('--zbinsize', default=0.05, type="float",
                          help='bin size in redshift space')
        parser.add_option('--finalzbinsize', default=0.0, type="float",
                          help='final bin size in redshift space for final z bin')
        parser.add_option('--nbins', default=25, type="float",
                          help='number of bins in log redshift space')
        parser.add_option('--equalbins', default=False, action="store_true",
                          help='if set, every bin contains the same number of SNe')
        parser.add_option('--corrzbins', default=False, action="store_true",
                          help='if set, run doSNBEAMS to get correlated measurements at log-spaced z control points')
        parser.add_option('--nzskip', default=0, type="int",
                          help='This is a hack - don\t write the first x bins to the output files')

        parser.add_option('-f','--fitresfile', default='ps1_psnidprob.fitres', type="string",
                          help='fitres file with the SN Ia data')
        parser.add_option('-o','--outfile', default='beamsCosmo.out', type="string",
                          help='Output file with the derived parameters for each redshift bin')

        parser.add_option('--mkplot', default=False, action="store_true",
                          help='plot the results')
        parser.add_option('--usefitresmu', default=False, action="store_true",
                          help='Use the distance modulus given in the fitres file')

        parser.add_option('-p','--paramfile', default='', type="string",
                          help='fitres file with the SN Ia data')
        parser.add_option('-p','--paramdefaultfile', default='BEAMS.params', type="string",
                          help='default parameter file to populate with SNCosmo options')
        parser.add_option('--showpriorprobs', default=False, action="store_true",
                          help='when plotting, show the prior probabilities instead of the (estimated) posterior probs.')


        parser.add_option('--covmatfile', default='BEAMS.covmat', type="string",
                          help='Output covariance matrix')

        parser.add_option('--mcsubset', default=False, action="store_true",
                          help='generate a random subset of SNe from the fitres file')
        parser.add_option('--subsetsize', default=125, type="int",
                          help='number of SNe in each MC subset ')
        parser.add_option('--nmc', default=100, type="int",
                          help='number of MC samples ')
        parser.add_option('--mclowz', default="", type="string",
                          help='low-z SNe, to be appended to the MC sample')

        parser.add_option('--onlyIa', default=False, action="store_true",
                          help='remove the TYPE != 1 SNe from the bunch')


        return(parser)

    def main(self,fitres,mkcuts=True):
        from txtobj import txtobj
        import cosmo

        fr = txtobj(fitres,fitresheader=True)
        fr.MU,fr.MUERR = salt2mu(x1=fr.x1,x1err=fr.x1ERR,c=fr.c,cerr=fr.cERR,mb=fr.mB,mberr=fr.mBERR,
                                 cov_x1_c=fr.COV_x1_c,cov_x1_x0=fr.COV_x1_x0,cov_c_x0=fr.COV_c_x0,
                                 alpha=self.options.salt2alpha,alphaerr=self.options.salt2alphaerr,
                                 beta=self.options.salt2beta,betaerr=self.options.salt2betaerr,
                                 x0=fr.x0,sigint=self.options.sigint,z=fr.zHD)

        if mkcuts:
            # Light curve cuts
            if self.options.x1ccircle:
                # I'm just going to assume cmax = abs(cmin) and same for x1
                cols = np.where((fr.x1**2./self.options.x1range[0]**2. + fr.c**2./self.options.crange[0]**2. < 1) &
                                (fr.x1ERR < self.options.x1errmax) & (fr.PKMJDERR < self.options.pkmjderrmax/(1+fr.zHD)) &
                                (fr.FITPROB > self.options.fitprobmin) &
                                (fr.z > self.options.zmin) & (fr.z < self.options.zmax) &
                                (fr.__dict__[self.options.piacol] >= 0))
            else:
                cols = np.where((fr.x1 > self.options.x1range[0]) & (fr.x1 < self.options.x1range[1]) &
                                (fr.c > self.options.crange[0]) & (fr.c < self.options.crange[1]) &
                                (fr.x1ERR < self.options.x1errmax) & (fr.PKMJDERR < self.options.pkmjderrmax) &
                                (fr.FITPROB > self.options.fitprobmin) &
                                (fr.z > self.options.zmin) & (fr.z < self.options.zmax) &
                                (fr.__dict__[self.options.piacol] >= 0))
            for k in fr.__dict__.keys():
                fr.__dict__[k] = fr.__dict__[k][cols]
        if self.options.onlyIa:
            cols = np.where(fr.TYPE == 1)
            for k in fr.__dict__.keys():
                fr.__dict__[k] = fr.__dict__[k][cols]

        # Prior SN Ia probabilities
        P_Ia = np.zeros(len(fr.CID))
        for i in range(len(fr.CID)):
            P_Ia[i] = fr.__dict__[self.options.piacol][i]
            if self.options.specconfcol:
                if fr.__dict__[self.options.specconfcol][i] == 1:
                    P_Ia[i] = 1

        if self.options.corrzbins:
            from doSNBEAMS import BEAMS
        else:
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

        if self.options.corrzbins:
            beam.options.zmin = self.options.zmin
            beam.options.zmax = self.options.zmax
            beam.options.nzbins = self.options.nbins

            # make the BEAMS input file
            fout = open('BEAMS_SN.input','w')
            print >> fout, '# PA z mu mu_err'
            for i in range(len(fr.MU)):
                if fr.zHD[i] > self.options.zmin and fr.zHD[i] <= self.options.zmax:
                    print >> fout, '%.3f %.4f %.4f %.4f'%(P_Ia[i],fr.zHD[i],fr.MU[i],fr.MUERR[i])
            fout.close()

            beam.options.append = False
            beam.options.clobber = self.options.clobber
            beam.options.outputfile = self.options.outfile
            beam.options.equalbins = self.options.equalbins
            beam.options.covmatfile = self.options.covmatfile
            beam.options.nzskip = self.options.nzskip
            beam.main(options.inputfile)
            bms = txtobj(self.options.outfile)
            self.writeBinCorrFitres('%s.fitres'%self.options.outfile.split('.')[0],bms,skip=self.options.nzskip,fr=fr)
            return

        # loop over the redshift bins
        append = False
        z = np.logspace(np.log10(self.options.zmin),np.log10(self.options.zmax),num=self.options.nbins)
        if self.options.equalbins:
            from scipy import stats
            z = stats.mstats.mquantiles(fr.zHD,np.arange(0,1,1./self.options.nbins))
            if self.options.finalzbinsize:
                z = np.append(z,self.options.zmax-self.options.finalzbinsize)
            z = np.append(z,self.options.zmax)
        os.system('rm %s'%self.options.outfile)
        for zmin,zmax in zip(z[:-1],z[1:]):
            if self.options.verbose: print('%.3f < z < %.3f'%(zmin,zmax))
            if zmin > self.options.zmin:
                append = True; clobber = False
            else: clobber = True

            # make the BEAMS input file
            fout = open('BEAMS_SN.input','w')
            print >> fout, '# PA resid resid_err'
            for i in range(len(fr.MU)):
                if fr.zHD[i] > zmin and fr.zHD[i] <= zmax:
                    print >> fout, '%.3f %.4f %.4f'%(P_Ia[i],fr.MU[i]-cosmo.mu(fr.zHD[i]),fr.MUERR[i])
            fout.close()

            beam.options.append = append
            beam.options.clobber = clobber
            beam.options.outputfile = self.options.outfile
            config = ConfigParser.ConfigParser()
            config.read(self.options.paramdefaultfile)
            for o in beam.options.__dict__.keys():
                config.set('all',o,str(beam.options.__dict__[o]).replace('(','').replace(')',''))
                if str(beam.options.__dict__[o]) == 'False':
                    config.set('all',o,0)
                elif str(beam.options.__dict__[o]) == 'True':
                    config.set('all',o,1)
    
            root = os.path.splitext(fitres)[0]
            fout = open('%s.params'%root,'w')
            config.write(fout); fout.close()
            os.system('./doBEAMS.py -p %s.params -o %s'%(root,self.options.outfile))

            #beam.main(options.inputfile)

        bms = txtobj(self.options.outfile)
        self.writeBinFitres('%s.fitres'%self.options.outfile.split('.')[0],bms,fr=fr)

    def writeBinFitres(self,outfile,bms,fr=None):
        import os,cosmo

        from txtobj import txtobj
        from iterstat import iterstat

        fout = open(outfile,'w')
        print >> fout, fitresheader

        z = np.logspace(np.log10(self.options.zmin),np.log10(self.options.zmax),num=self.options.nbins)
        if self.options.equalbins:
            from scipy import stats
            z = stats.mstats.mquantiles(fr.zHD,np.arange(0,1,1./self.options.nbins))
            if self.options.finalzbinsize:
                z = np.append(z,self.options.zmax-self.options.finalzbinsize)
            z = np.append(z,self.options.zmax)
        for zmin,zmax,i in zip(z[:-1],z[1:],range(len(z[:-1]))):

            #cols = np.where((fr.zHD > zmin) & (fr.zHD < zmax) & (fr.TYPE == 1))[0]
            #from doSNBEAMS import weighted_avg_and_std
            #md,std = weighted_avg_and_std(fr.MU[cols]-cosmo.mu(fr.zHD[cols]),1/fr.MUERR[cols]**2.)
            #std = 1/np.sqrt(np.sum(1/fr.MUERR[cols]**2.))

            outvars = ()
            for v in fitresvars:
                if v == 'zHD':
                    outvars += ((zmin+zmax)/2.,)
                elif v == 'z':
                    outvars += ((zmin+zmax)/2.,)
                elif v == 'mB':
                    outvars += (bms.muA[i]+cosmo.mu((zmin+zmax)/2.)-19.3,)#(md+cosmo.mu((zmin+zmax)/2.)-19.3,)
                elif v == 'mBERR':
                    outvars += (bms.muAerr[i],)#(std,)
                else:
                    outvars += (0,)
            print >> fout, fitresfmt%outvars

    def writeBinCorrFitres(self,outfile,bms,skip=0,fr=None):
        import os,cosmo

        from txtobj import txtobj
        from iterstat import iterstat

        fout = open(outfile,'w')
        print >> fout, fitresheader
        z = np.logspace(np.log10(self.options.zmin),np.log10(self.options.zmax),num=self.options.nbins)
        if self.options.equalbins:
            from scipy import stats
            z = stats.mstats.mquantiles(fr.zHD,np.arange(0,1,1./self.options.nbins))

        for zcntrl,i in zip(z[skip:],range(len(z[skip:]))):

            outvars = ()
            for v in fitresvars:
                if v == 'zHD':
                    outvars += (zcntrl,)
                elif v == 'z':
                    outvars += (zcntrl,)
                elif v == 'mB':
                    outvars += (bms.muA[i]-19.3,)
                elif v == 'mBERR':
                    outvars += (bms.muAerr[i],)
                else:
                    outvars += (0,)
            print >> fout, fitresfmt%outvars


    def mkplot(self,fitresfile=None,showpriorprobs=False):
        
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
                            (fr.z > self.options.zmin) & (fr.z < self.options.zmax))
        else:
            cols = np.where((fr.x1 > self.options.x1range[0]) & (fr.x1 < self.options.x1range[1]) &
                            (fr.c > self.options.crange[0]) & (fr.c < self.options.crange[1]) &
                            (fr.x1ERR < self.options.x1errmax) & (fr.PKMJDERR < self.options.pkmjderrmax) &
                            (fr.FITPROB > self.options.fitprobmin) &
                            (fr.z > self.options.zmin) & (fr.z < self.options.zmax))
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

        if not self.options.usefitresmu:
            mu,muerr = salt2mu(x1=fr.x1,x1err=fr.x1ERR,c=fr.c,cerr=fr.cERR,mb=fr.mB,mberr=fr.mBERR,
                               cov_x1_c=fr.COV_x1_c,cov_x1_x0=fr.COV_x1_x0,cov_c_x0=fr.COV_c_x0,
                               alpha=self.options.salt2alpha,alphaerr=self.options.salt2alphaerr,
                               beta=self.options.salt2beta,betaerr=self.options.salt2betaerr,
                               x0=fr.x0,sigint=self.options.sigint,z=fr.zHD)
            fr.MU,fr.MUERR = mu,muerr
        else: mu,muerr = fr.MU[cols],fr.MUERR[cols]
        mures = mu - cosmo.mu(fr.zHD)

        mu_cc_p50,mu_ia_p50,z_ia_p50,z_cc_p50 = np.array([]),np.array([]),np.array([]),np.array([])
        for zmin,zmax,i in zip(np.arange(0,0.7,0.05),np.arange(0.05,0.75,0.05),range(len(np.arange(0.05,0.75,0.05)))):
            cols = np.where((fr.z > zmin) & (fr.z < zmax))
            zbin = np.mean([zmin,zmax])

            P_Ia_post = gauss(mu-cosmo.mu(fr.zHD),bms.muA[i],np.sqrt(muerr**2.+bms.sigA[i]**2.))
            P_CC_post = gauss(mu-cosmo.mu(fr.zHD),bms.muB[i],np.sqrt(muerr**2.+bms.sigB[i]**2.))
            if not showpriorprobs:
                P_Ia_final = P_Ia_post/(P_Ia_post+P_CC_post)
            else:
                P_Ia_final = fr.__dict__[self.options.piacol]
            mu_cc_p50 = np.append(mu_cc_p50,mures[cols][np.where(P_Ia_final[cols] < 0.5)])
            mu_ia_p50 = np.append(mu_ia_p50,mures[cols][np.where(P_Ia_final[cols] > 0.5)])
            z_cc_p50 = np.append(z_cc_p50,fr.zHD[cols][np.where(P_Ia_final[cols] < 0.5)])
            z_ia_p50 = np.append(z_ia_p50,fr.zHD[cols][np.where(P_Ia_final[cols] > 0.5)])


            ax1.errorbar(fr.zHD[cols],mu[cols],yerr=muerr[cols],fmt='o',color='0.5',alpha=0.5)
            sc = ax1.scatter(fr.zHD[cols],mu[cols],c=P_Ia_final[cols],s=30,zorder=9,cmap='RdBu_r',vmin=0.0,vmax=1.0,alpha=0.6)
            ax2.errorbar(fr.zHD[cols],mu[cols]-cosmo.mu(fr.zHD[cols]),yerr=muerr[cols],fmt='o',color='0.5',alpha=0.5)
            sc = ax2.scatter(fr.zHD[cols],mu[cols]-cosmo.mu(fr.zHD[cols]),c=P_Ia_final[cols],s=30,zorder=9,cmap='RdBu_r',vmin=0.0,vmax=1.0,alpha=0.6)

            ax1.errorbar((zmin+zmax)/2.,bms.muA[i]+cosmo.mu(zbin),yerr=[[bms.muAerr_m[i]],[bms.muAerr_p[i]]],fmt='o',color='lightgreen',zorder=40,lw=3,ms=5,label='$\mu_{Ia}$')
            ax1.errorbar((zmin+zmax)/2.,bms.muB[i]+cosmo.mu(zbin),yerr=[[bms.muBerr_m[i]],[bms.muBerr_p[i]]],fmt='o',color='b',zorder=40,lw=3,ms=5,label='$\mu_{CC}$')
            ax2.errorbar((zmin+zmax)/2.,bms.muA[i],yerr=[[bms.muAerr_m[i]],[bms.muAerr_p[i]]],fmt='o',color='lightgreen',zorder=40,lw=3,ms=5)
            ax2.errorbar((zmin+zmax)/2.,bms.muB[i],yerr=[[bms.muBerr_m[i]],[bms.muBerr_p[i]]],fmt='o',color='b',zorder=40,lw=3,ms=5)
            if i == 0:
                ax1.legend(numpoints=1,loc='upper left')


        ax1.set_xlim([0.02,0.79])
        ax2.set_xlim([0.02,0.79])
        ax1.set_ylim([35,45])
        ax2.set_ylim([-3,3])

        cax = plt.axes([0.55,0.525,0.35,0.06])
        cb = plt.colorbar(sc,cax=cax,orientation='horizontal',ticks=[0,0.2,0.4,0.6,0.8,1.0])
        cax.set_xlabel('$P(Ia)$',labelpad=-65,fontsize='large')

        n_mu = np.histogram(mu_cc_p50,bins=30,range=[-3,3])
        ax3.plot(n_mu[0]/float(len(fr.MU)),n_mu[1][:-1],color='b',drawstyle='steps',lw=2)
        n_mu = np.histogram(mu_ia_p50,bins=30,range=[-3,3])
        ax3.plot(n_mu[0]/float(len(fr.MU)),n_mu[1][:-1],color='lightgreen',drawstyle='steps',lw=2)

        ax1.set_xlabel('$z$',fontsize=20)
        ax1.set_ylabel('$\mu$',fontsize=20)
        ax2.set_xlabel('$z$',fontsize=20)
        ax2.set_ylabel('$\mu - \mu_{\Lambda CDM}$',fontsize=20)
        ax3.xaxis.set_ticks([])

        import pdb; pdb.set_trace()

    def mcsamp(self,fitresfile,mciter,lowzfile,nsne):
        import os
        from txtobj import txtobj
        import numpy as np

        fitresheader = """# VERSION: PS1_PS1MD
# FITOPT:  NONE
# ---------------------------------------- 
NVAR: 31 
VARNAMES:  CID IDSURVEY TYPE FIELD zHD zHDERR z zERR HOST_LOGMASS HOST_LOGMASS_ERR SNRMAX1 SNRMAX2 SNRMAX3 PKMJD PKMJDERR x1 x1ERR c cERR mB mBERR x0 x0ERR COV_x1_c COV_x1_x0 COV_c_x0 NDOF FITCHI2 FITPROB PBAYES_Ia PGAL_Ia PFITPROB_Ia
# VERSION_SNANA      = v10_39i 
# VERSION_PHOTOMETRY = PS1_PS1MD 
# TABLE NAME: FITRES 
# 
"""
        fitresvars = ["CID","IDSURVEY","TYPE","FIELD",
                      "zHD","zHDERR","z","zERR","HOST_LOGMASS",
                      "HOST_LOGMASS_ERR","SNRMAX1","SNRMAX2",
                      "SNRMAX3","PKMJD","PKMJDERR","x1","x1ERR",
                      "c","cERR","mB","mBERR","x0","x0ERR","COV_x1_c",
                      "COV_x1_x0","COV_c_x0","NDOF","FITCHI2","FITPROB",
                      "PBAYES_Ia","PGAL_Ia","PFITPROB_Ia"]
        fitresfmt = 'SN: %s %i %i %s %.5f %.5f %.5f %.5f %i %i %.4f %.4f %.4f %.3f %.3f %8.5e %8.5e %8.5e %8.5e %.4f %.4f %8.5e %8.5e %8.5e %8.5e %8.5e %i %.4f %.4f %.4f %.4f %.4f'

        name,ext = os.path.splitext(fitresfile)
        fitresoutfile = '%s_mc%i%s'%(name,mciter,ext)

        fr = txtobj(fitresfile,fitresheader=True)
        frlowz = txtobj(lowzfile,fitresheader=True)    
        # Light curve cuts
        if self.options.x1ccircle:
            # I'm just going to assume cmax = abs(cmin) and same for x1
            cols = np.where((fr.x1**2./self.options.x1range[0]**2. + fr.c**2./self.options.crange[0]**2. < 1) &
                            (fr.x1ERR < self.options.x1errmax) & (fr.PKMJDERR < self.options.pkmjderrmax/(1+fr.zHD)) &
                            (fr.FITPROB > self.options.fitprobmin) &
                            (fr.z > self.options.zmin) & (fr.z < self.options.zmax) &
                            (fr.__dict__[self.options.piacol] >= 0))
        else:
            cols = np.where((fr.x1 > self.options.x1range[0]) & (fr.x1 < self.options.x1range[1]) &
                            (fr.c > self.options.crange[0]) & (fr.c < self.options.crange[1]) &
                            (fr.x1ERR < self.options.x1errmax) & (fr.PKMJDERR < self.options.pkmjderrmax) &
                            (fr.FITPROB > self.options.fitprobmin) &
                            (fr.z > self.options.zmin) & (fr.z < self.options.zmax) &
                            (fr.__dict__[self.options.piacol] >= 0))
        for k in fr.__dict__.keys():
            fr.__dict__[k] = fr.__dict__[k][cols]



        writefitres(fr,np.random.randint(len(fr.CID),size=nsne),fitresoutfile,fitresheader=fitresheader,
                    fitresvars=fitresvars,fitresfmt=fitresfmt)
        writefitres(frlowz,range(len(frlowz.CID)),fitresoutfile,append=True,fitresheader=fitresheader,
                    fitresvars=fitresvars,fitresfmt=fitresfmt)

        return(fitresoutfile)


def gauss(x,x0,sigma):
    return(normpdf(x,x0,sigma))

def normpdf(x, mu, sigma):
    u = (x-mu)/np.abs(sigma)
    y = (1/(np.sqrt(2*np.pi)*np.abs(sigma)))*np.exp(-u*u/2)
    return y
        
def gausshist(x,sigma=1,peak=1.,center=0):

    y = peak*np.exp(-(x-center)**2./(2.*sigma**2.))

    return(y)

def salt2mu(x1=None,x1err=None,
            c=None,cerr=None,
            mb=None,mberr=None,
            cov_x1_c=None,cov_x1_x0=None,cov_c_x0=None,
            alpha=None,beta=None,
            alphaerr=None,betaerr=None,
            M=None,x0=None,sigint=None,z=None,peczerr=0.0005):
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
        if sigint: mu = mu + ufloat(0,sigint)
        zerr = peczerr*5.0/np.log(10)*(1.0+z[i])/(z[i]*(1.0+z[i]/2.0))

        mu = mu + ufloat(0,np.sqrt(zerr**2. + 0.055**2.*z[i]**2.))
        mu_out,muerr_out = np.append(mu_out,mu.n),np.append(muerr_out,mu.std_dev)

    return(mu_out,muerr_out)

def writefitres(fitresobj,cols,outfile,append=False,fitresheader=None,
                fitresvars=None,fitresfmt=None):
    import os
    if not append:
        fout = open(outfile,'w')
        print >> fout, fitresheader
    else:
        fout = open(outfile,'a')

    for c in cols:
        outvars = ()
        for v in fitresvars:
            outvars += (fitresobj.__dict__[v][c],)
        print >> fout, fitresfmt%outvars

    fout.close()                

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

    if options.mcsubset:
        outfile_orig = options.outfile[:]
        for i in range(options.nmc):
            frfile = sne.mcsamp(options.fitresfile,i,options.mclowz,options.subsetsize)
            name,ext = os.path.splitext(outfile_orig)
            options.outfile = '%s_mc%i%s'%(name,i,ext)
            sne.main(frfile,mkcuts=False)
    if not options.mkplot:
        sne.main(options.fitresfile)
    else:
        sne.mkplot(fitresfile=options.fitresfile,showpriorprobs=options.showpriorprobs)

