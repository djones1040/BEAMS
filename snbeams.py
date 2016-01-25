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
fitresheaderbeams = """# CID IDSURVEY TYPE FIELD zHD zHDERR z zERR HOST_LOGMASS HOST_LOGMASS_ERR SNRMAX1 SNRMAX2 SNRMAX3 PKMJD PKMJDERR x1 x1ERR c cERR mB mBERR x0 x0ERR COV_x1_c COV_x1_x0 COV_c_x0 NDOF FITCHI2 FITPROB PA PL
"""
fitresfmtbeams = '%s %i %i %s %.5f %.5f %.5f %.5f %.4f %.4f %.4f %.4f %.4f %.3f %.3f %8.5e %8.5e %8.5e %8.5e %.4f %.4f %8.5e %8.5e %8.5e %8.5e %8.5e %i %.4f %.4f %.4f %.4f'
fitresvarsbeams = ["CID","IDSURVEY","TYPE","FIELD",
                   "zHD","zHDERR","z","zERR","HOST_LOGMASS",
                   "HOST_LOGMASS_ERR","SNRMAX1","SNRMAX2",
                   "SNRMAX3","PKMJD","PKMJDERR","x1","x1ERR",
                   "c","cERR","mB","mBERR","x0","x0ERR","COV_x1_c",
                   "COV_x1_x0","COV_c_x0","NDOF","FITCHI2","FITPROB",
                   "PA","PL"]


fitresvars = ["CID","IDSURVEY","TYPE","FIELD",
              "zHD","zHDERR","z","zERR","HOST_LOGMASS",
              "HOST_LOGMASS_ERR","SNRMAX1","SNRMAX2",
              "SNRMAX3","PKMJD","PKMJDERR","x1","x1ERR",
              "c","cERR","mB","mBERR","x0","x0ERR","COV_x1_c",
              "COV_x1_x0","COV_c_x0","NDOF","FITCHI2","FITPROB"]
fitresfmt = 'SN: %s %i %i %s %.5f %.5f %.5f %.5f %.4f %.4f %.4f %.4f %.4f %.3f %.3f %8.5e %8.5e %8.5e %8.5e %.4f %.4f %8.5e %8.5e %8.5e %8.5e %8.5e %i %.4f %.4f'

class snbeams:
    def __init__(self):
        self.clobber = False
        self.verbose = False

    def add_options(self, parser=None, usage=None, config=None):
        import optparse
        if parser == None:
            parser = optparse.OptionParser(usage=usage, conflict_handler="resolve")

        # the basics
        parser.add_option('-v', '--verbose', action="count", dest="verbose",default=1)
        parser.add_option('--debug', default=False, action="store_true",
                          help='debug mode: more output and debug files')
        parser.add_option('--clobber', default=False, action="store_true",
                          help='clobber output image')

        if config:
            parser.add_option('--piacol', default=config.get('main','piacol'), type="string",
                              help='Column in FITRES file used as guess at P(Ia)')
            parser.add_option('--specconfcol', default=config.get('main','specconfcol'), type="string",
                              help='Column in FITRES file indicating spec.-confirmed SNe with 1')

            # Light curve cut parameters
            parser.add_option(
                '--crange', default=default=map(float,config.get('main','crange').split(',')),type="float",
                help='Peculiar velocity error (default=%default)',nargs=2)
            parser.add_option(
                '--x1range', default=map(float,config.get('main','crange').split(',')),type="float",
                help='Peculiar velocity error (default=%default)',nargs=2)
            parser.add_option('--x1cellipse',default=config.get('main','x1cellipse'),action="store_true",
                          help='Elliptical, not box, cut in x1 and c')
            parser.add_option(
                '--fitprobmin', default=config.get('main','fitprobmin'),type="float",
                help='Peculiar velocity error (default=%default)')
            parser.add_option(
                '--x1errmax', default=config.get('main','x1errmax'),type="float",
                help='Peculiar velocity error (default=%default)')
            parser.add_option(
                '--pkmjderrmax', default=config.get('main','pkmjderrmax'),type="float",
                help='Peculiar velocity error (default=%default)')

            # SALT2 parameters and intrinsic dispersion
            parser.add_option('--salt2alpha', default=config.get('main','salt2alpha'), type="float",
                              help='SALT2 alpha parameter from a spectroscopic sample (default=%default)')
            parser.add_option('--salt2alphaerr', default=config.get('main','salt2alphaerr'), type="float",
                              help='nominal SALT2 alpha uncertainty from a spectroscopic sample (default=%default)')
            parser.add_option('--salt2beta', default=config.get('main','salt2beta'), type="float",
                              help='nominal SALT2 beta parameter from a spec. sample (default=%default)')
            parser.add_option('--salt2betaerr', default=config.get('main','salt2betaerr'), type="float",
                              help='nominal SALT2 beta uncertainty from a spec. sample (default=%default)')
            parser.add_option('--sigint', default=config.get('main','sigint'), type="float",
                              help='nominal intrinsic dispersion, MCMC fits for this if not specified (default=%default)')

            # Mass options
            parser.add_option(
                '--masscorr', default=config.get('main','masscorr'),action="store_true",
                help='If true, perform mass correction (default=%default)')
            parser.add_option(
                '--masscorrfixed', default=config.get('main','masscorrfixed'),action="store_true",
                help='If true, perform fixed mass correction (default=%default)')
            parser.add_option(
                '--masscorrmag', default=(0.07,0.023),type="float",
                help="""mass corr. and uncertainty (default=%default)""",nargs=2)

            parser.add_option('--nthreads', default=config.get('main','nthreads'), type="int",
                              help='Number of threads for MCMC')
            parser.add_option('--zmin', default=config.get('main','zmin'), type="float",
                              help='minimum redshift')
            parser.add_option('--zmax', default=config.get('main','zmax'), type="float",
                              help='maximum redshift')
            parser.add_option('--nbins', default=config.get('main','nbins'), type="float",
                              help='number of bins in log redshift space')
            parser.add_option('--equalbins', default=config.get('main','equalbins'), action="store_true",
                              help='if set, every bin contains the same number of SNe')
            parser.add_option('--snpars', default=config.get('main','snpars'), action="store_true",
                              help='if set, marginalize over alpha and beta for SNe Ia (CC SNe set to priors)')


            parser.add_option('-f','--fitresfile', default=config.get('main','fitresfile'), type="string",
                              help='fitres file with the SN Ia data')
            parser.add_option('-o','--outfile', default=config.get('main','outfile'), type="string",
                              help='Output file with the derived parameters for each redshift bin')

            parser.add_option('--usefitresmu', default=config.get('main','usefitresmu'), action="store_true",
                              help='Use the distance modulus given in the fitres file')

            parser.add_option('-p','--paramdefaultfile', default=config.get('main','paramdefaultfile'), type="string",
                              help='default parameter file to populate with SNCosmo options')

            parser.add_option('--mcsubset', default=config.get('main','mcsubset'), action="store_true",
                              help='generate a random subset of SNe from the fitres file')
            parser.add_option('--subsetsize', default=config.get('main','subsetsize'), type="int",
                              help='number of SNe in each MC subset ')
            parser.add_option('--nmc', default=config.get('main','nmc'), type="int",
                              help='number of MC samples ')
            parser.add_option('--mclowz', default=config.get('main','mclowz'), type="string",
                              help='low-z SNe, to be appended to the MC sample')

            parser.add_option('--onlyIa', default=config.get('main','onlyIa'), action="store_true",
                              help='remove the TYPE != 1 SNe from the bunch')

            # alternate functional models
            parser.add_option('--twogauss', default=config.get('main','twogauss'), action="store_true",
                              help='two gaussians for pop. B')
            parser.add_option('--skewedgauss', default=config.get('main','skewedgauss'), action="store_true",
                              help='skewed gaussian for pop. B')
            parser.add_option('--simcc', default=config.get('main','simcc'), type='string',
                              help='if filename is given, construct a polynomial-altered empirical CC SN function')

        else:
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
            parser.add_option('--x1cellipse',default=False,action="store_true",
                              help='Circle cut in x1 and c')
            parser.add_option(
                '--fitprobmin', default=0.0,type="float",
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
            parser.add_option('--sigint', default=None, type="float",
                              help='nominal intrinsic dispersion, MCMC fits for this if not specified (default=%default)')

            # Mass options
            parser.add_option(
                '--masscorr', default=False,action="store_true",
                help='If true, perform mass correction (default=%default)')
            parser.add_option(
                '--masscorrfixed', default=False,action="store_true",
                help='If true, perform fixed mass correction (default=%default)')
            parser.add_option(
                '--masscorrmag', default=(0.07,0.023),type="float",
                help="""mass corr. and uncertainty (default=%default)""",nargs=2)
            
            parser.add_option('--nthreads', default=8, type="int",
                              help='Number of threads for MCMC')
            parser.add_option('--zmin', default=0.01, type="float",
                              help='minimum redshift')
            parser.add_option('--zmax', default=0.75, type="float",
                              help='maximum redshift')
            parser.add_option('--nbins', default=25, type="float",
                              help='number of bins in log redshift space')
            parser.add_option('--equalbins', default=False, action="store_true",
                              help='if set, every bin contains the same number of SNe')
            parser.add_option('--snpars', default=False, action="store_true",
                              help='if set, marginalize over alpha and beta for SNe Ia (CC SNe set to priors)')
            
            
            parser.add_option('-f','--fitresfile', default='ps1_psnidprob.fitres', type="string",
                              help='fitres file with the SN Ia data')
            parser.add_option('-o','--outfile', default='beamsCosmo.out', type="string",
                              help='Output file with the derived parameters for each redshift bin')
            
            parser.add_option('--usefitresmu', default=False, action="store_true",
                              help='Use the distance modulus given in the fitres file')
            
            parser.add_option('-p','--paramdefaultfile', default='BEAMS.params', type="string",
                              help='default parameter file to populate with SNCosmo options')
            
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

            # alternate functional models
            parser.add_option('--twogauss', default=False, action="store_true",
                              help='two gaussians for pop. B')
            parser.add_option('--skewedgauss', default=False, action="store_true",
                              help='skewed gaussian for pop. B')
            parser.add_option('--simcc', default='', type='string',
                              help='if filename is given, construct a polynomial-altered empirical CC SN function')
            
        parser.add_option('-p','--paramfile', default='', type="string",
                          help='fitres file with the SN Ia data')

        return(parser)

    def main(self,fitres,mkcuts=True):
        from txtobj import txtobj
        from astropy.cosmology import Planck13 as cosmo

        fr = txtobj(fitres,fitresheader=True)
        if self.options.simcc:
            simcc = txtobj(self.options.simcc,fitresheader=True)
        
        from doSNBEAMS import salt2mu_aberr
        fr.MU,fr.MUERR = salt2mu_aberr(x1=fr.x1,x1err=fr.x1ERR,c=fr.c,cerr=fr.cERR,mb=fr.mB,mberr=fr.mBERR,
                                       cov_x1_c=fr.COV_x1_c,cov_x1_x0=fr.COV_x1_x0,cov_c_x0=fr.COV_c_x0,
                                       alpha=self.options.salt2alpha,alphaerr=self.options.salt2alphaerr,
                                       beta=self.options.salt2beta,betaerr=self.options.salt2betaerr,
                                       x0=fr.x0,sigint=self.options.sigint,z=fr.zHD)
        if self.options.simcc:
            simcc.MU,simcc.MUERR = salt2mu_aberr(x1=simcc.x1,x1err=simcc.x1ERR,c=simcc.c,cerr=simcc.cERR,mb=simcc.mB,mberr=simcc.mBERR,
                                                 cov_x1_c=simcc.COV_x1_c,cov_x1_x0=simcc.COV_x1_x0,cov_c_x0=simcc.COV_c_x0,
                                                 alpha=self.options.salt2alpha,alphaerr=self.options.salt2alphaerr,
                                                 beta=self.options.salt2beta,betaerr=self.options.salt2betaerr,
                                                 x0=simcc.x0,sigint=self.options.sigint,z=simcc.zHD)

        if mkcuts:
            # Light curve cuts
            sf = -2.5/(fr.x0*np.log(10.0))
            invvars = 1./(fr.mBERR**2.+ self.options.salt2alpha**2. * fr.x1ERR**2. + \
                              self.options.salt2beta**2. * fr.cERR**2. +  2.0 * self.options.salt2alpha * (fr.COV_x1_x0*sf) - \
                              2.0 * self.options.salt2beta * (fr.COV_c_x0*sf) - \
                              2.0 * self.options.salt2alpha*self.options.salt2beta * (fr.COV_x1_c) )
            if self.options.x1cellipse:
                # I'm just going to assume cmax = abs(cmin) and same for x1
                cols = np.where((fr.x1**2./self.options.x1range[0]**2. + fr.c**2./self.options.crange[0]**2. < 1) &
                                (fr.x1ERR < self.options.x1errmax) & (fr.PKMJDERR < self.options.pkmjderrmax/(1+fr.zHD)) &
                                (fr.FITPROB >= self.options.fitprobmin) &
                                (fr.z > self.options.zmin) & (fr.z < self.options.zmax) &
                                (fr.__dict__[self.options.piacol] >= 0) & (invvars > 0))
            else:
                cols = np.where((fr.x1 > self.options.x1range[0]) & (fr.x1 < self.options.x1range[1]) &
                                (fr.c > self.options.crange[0]) & (fr.c < self.options.crange[1]) &
                                (fr.x1ERR < self.options.x1errmax) & (fr.PKMJDERR < self.options.pkmjderrmax) &
                                (fr.FITPROB >= self.options.fitprobmin) &
                                (fr.z > self.options.zmin) & (fr.z < self.options.zmax) &
                                (fr.__dict__[self.options.piacol] >= 0) & (invvars > 0))

            for k in fr.__dict__.keys():
                fr.__dict__[k] = fr.__dict__[k][cols]
            if self.options.simcc:
                # Light curve cuts
                sf = -2.5/(simcc.x0*np.log(10.0))
                invvars = 1./(simcc.mBERR**2.+ self.options.salt2alpha**2. * simcc.x1ERR**2. + \
                                  self.options.salt2beta**2. * simcc.cERR**2. +  2.0 * self.options.salt2alpha * (simcc.COV_x1_x0*sf) - \
                                  2.0 * self.options.salt2beta * (simcc.COV_c_x0*sf) - \
                                  2.0 * self.options.salt2alpha*self.options.salt2beta * (simcc.COV_x1_c) )
                if self.options.x1cellipse:
                    # I'm just going to assume cmax = abs(cmin) and same for x1
                    cols = np.where((simcc.x1**2./self.options.x1range[0]**2. + simcc.c**2./self.options.crange[0]**2. < 1) &
                                    (simcc.x1ERR < self.options.x1errmax) & (simcc.PKMJDERR < self.options.pkmjderrmax/(1+simcc.zHD)) &
                                    (simcc.FITPROB >= self.options.fitprobmin) &
                                    (simcc.z > self.options.zmin) & (simcc.z < self.options.zmax) &
                                    (simcc.__dict__[self.options.piacol] >= 0) & (invvars > 0))
                else:
                    cols = np.where((simcc.x1 > self.options.x1range[0]) & (simcc.x1 < self.options.x1range[1]) &
                                    (simcc.c > self.options.crange[0]) & (simcc.c < self.options.crange[1]) &
                                    (simcc.x1ERR < self.options.x1errmax) & (simcc.PKMJDERR < self.options.pkmjderrmax) &
                                    (simcc.FITPROB >= self.options.fitprobmin) &
                                    (simcc.z > self.options.zmin) & (simcc.z < self.options.zmax) &
                                    (simcc.__dict__[self.options.piacol] >= 0) & (invvars > 0))

                for k in simcc.__dict__.keys():
                    simcc.__dict__[k] = simcc.__dict__[k][cols]
        if self.options.onlyIa:
            cols = np.where(fr.TYPE == 1)
            for k in fr.__dict__.keys():
                fr.__dict__[k] = fr.__dict__[k][cols]

        root = os.path.splitext(fitres)[0]
        
        # Prior SN Ia probabilities
        P_Ia = np.zeros(len(fr.CID))
        for i in range(len(fr.CID)):
            P_Ia[i] = fr.__dict__[self.options.piacol][i]
            if self.options.specconfcol:
                if fr.__dict__[self.options.specconfcol][i] == 1:
                    P_Ia[i] = 1

        from doSNBEAMS import BEAMS
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
        beam.options.twogauss = self.options.twogauss
        beam.options.skewedgauss = self.options.skewedgauss
        beam.options.snpars = self.options.snpars

        beam.transformOptions()
        options.inputfile = '%s.input'%root
        if self.options.masscorr:
            beam.options.lstep = True
            beam.options.plcol = 'PL'
            import scipy.stats
            cols = np.where(fr.HOST_LOGMASS > 0)
            for k in fr.__dict__.keys():
                fr.__dict__[k] = fr.__dict__[k][cols]
            fr.PL = np.zeros(len(fr.CID))
            for i in range(len(fr.CID)):
                if fr.HOST_LOGMASS_ERR[i] == 0: fr.HOST_LOGMASS_ERR[i] = 1e-5
                fr.PL[i] = scipy.stats.norm.cdf(10,fr.HOST_LOGMASS[i],fr.HOST_LOGMASS_ERR[i])
            P_Ia = P_Ia[cols]            

        if self.options.masscorrfixed: beam.options.lstepfixed = True

        beam.options.zmin = self.options.zmin
        beam.options.zmax = self.options.zmax
        beam.options.nzbins = self.options.nbins

        # make the BEAMS input file
        fout = open('%s.input'%root,'w')
        fr.PA = fr.__dict__[self.options.piacol]
        if not self.options.masscorr: fr.PL = np.zeros(len(fr.PA))
        writefitres(fr,range(len(fr.PA)),'%s.input'%root,
                    fitresheader=fitresheaderbeams,
                    fitresfmt=fitresfmtbeams,
                    fitresvars=fitresvarsbeams)
        # make the sim. CC input file
        if self.options.simcc:
            writefitres(simcc,range(len(simcc.CID)),'%s.simcc.input'%root,
                        fitresheader=fitresheaderbeams,
                        fitresfmt=fitresfmtbeams,
                        fitresvars=fitresvarsbeams)
            beams.options.simcc = '%s.simcc.input'%root

        beam.options.append = False
        beam.options.clobber = self.options.clobber
        beam.options.outputfile = self.options.outfile
        beam.options.equalbins = self.options.equalbins
        beam.main(options.inputfile)
        bms = txtobj(self.options.outfile)
        self.writeBinCorrFitres('%s.fitres'%self.options.outfile.split('.')[0],bms,fr=fr)
        return

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
        fitresfmt = 'SN: %s %i %i %s %.5f %.5f %.5f %.5f %.4f %.4f %.4f %.4f %.4f %.3f %.3f %8.5e %8.5e %8.5e %8.5e %.4f %.4f %8.5e %8.5e %8.5e %8.5e %8.5e %i %.4f %.4f %.4f %.4f %.4f'

        name,ext = os.path.splitext(fitresfile)
        fitresoutfile = '%s_mc%i%s'%(name,mciter,ext)

        fr = txtobj(fitresfile,fitresheader=True)
        frlowz = txtobj(lowzfile,fitresheader=True)    
        # Light curve cuts
        if self.options.x1cellipse:
            # I'm just going to assume cmax = abs(cmin) and same for x1
            cols = np.where((fr.x1**2./self.options.x1range[0]**2. + fr.c**2./self.options.crange[0]**2. < 1) &
                            (fr.x1ERR < self.options.x1errmax) & (fr.PKMJDERR < self.options.pkmjderrmax/(1+fr.zHD)) &
                            (fr.FITPROB >= self.options.fitprobmin) &
                            (fr.z > self.options.zmin) & (fr.z < self.options.zmax) &
                            (fr.__dict__[self.options.piacol] >= 0))
        else:
            cols = np.where((fr.x1 > self.options.x1range[0]) & (fr.x1 < self.options.x1range[1]) &
                            (fr.c > self.options.crange[0]) & (fr.c < self.options.crange[1]) &
                            (fr.x1ERR < self.options.x1errmax) & (fr.PKMJDERR < self.options.pkmjderrmax) &
                            (fr.FITPROB >= self.options.fitprobmin) &
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
    alphatmp,betatmp = alpha,beta
    alpha,beta = ufloat(alpha,alphaerr),ufloat(beta,betaerr)

    sf = -2.5/(x0*np.log(10.0))
    cov_mb_c = cov_c_x0*sf
    cov_mb_x1 = cov_x1_x0*sf
    invvars = 1.0 / (mberr**2.+ alphatmp**2. * x1err**2. + betatmp**2. * cerr**2. + \
                         2.0 * alphatmp * (cov_x1_x0*sf) - 2.0 * betatmp * (cov_c_x0*sf) - \
                         2.0 * alphatmp*betatmp * (cov_x1_c) )

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

USAGE: snbeams.py [options]

examples:
"""

    import exceptions
    import os
    import optparse

    sne = snbeams()

    parser = sne.add_options(usage=usagestring)
    options,  args = parser.parse_args()
    if options.paramfile:
        config = ConfigParser.ConfigParser()
        config.read(options.paramfile)
    else: config=None
    parser = beam.add_options(usage=usagestring,config=config)
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
    elif not options.mkplot:
        sne.main(options.fitresfile)
    else:
        sne.mkplot(fitresfile=options.fitresfile,showpriorprobs=options.showpriorprobs)

