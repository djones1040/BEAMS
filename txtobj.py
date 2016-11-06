#!/usr/bin/env python
# D. Jones - 5/14/14

import glob
import os
import numpy as np
import exceptions
import pyfits

class txtobj:
    def __init__(self,filename,allstring=False,
                 cmpheader=False,sexheader=False,
                 useloadtxt=True,fitresheader=False,
                 delimiter=' ',skiprows=0,tabsep=False):
        if cmpheader: hdr = pyfits.getheader(filename)
        if fitresheader: skiprows=6

        coldefs = np.array([])
        if cmpheader:
            for k,v in zip(hdr.keys(),hdr.values()):
                if 'COLTBL' in k and k != 'NCOLTBL':
                    coldefs = np.append(coldefs,v)
        elif sexheader:
            fin = open(filename,'r')
            lines = fin.readlines()
            for l in lines:
                if l.startswith('#'):
                    coldefs = np.append(coldefs,filter(None,l.split(' '))[2])
        elif fitresheader:
            fin = open(filename,'r')
            lines = fin.readlines()
            for l in lines:
                if l.startswith('VARNAMES:'):
                    l = l.replace('\n','')
                    coldefs = np.array(filter(None,l.split(' ')))
                    coldefs = coldefs[np.where((coldefs != 'VARNAMES:') & (coldefs != '\n') & (coldefs != '#'))]
                    break
        else:
            fin = open(filename,'r')
            lines = fin.readlines()
            if not tabsep:
                coldefs = np.array(filter(None,lines[0].split(delimiter)))
                coldefs = coldefs[np.where(coldefs != '#')]
            else:
                l = lines[0].replace('\n','')
                coldefs = np.array(filter(None,l.split('\t')))
                coldefs = coldefs[np.where(coldefs != '#')]
        for i in range(len(coldefs)):
            coldefs[i] = coldefs[i].replace('\n','').replace('\t','').replace(' ','')
            if coldefs[i]:
                self.__dict__[coldefs[i]] = np.array([])

        self.filename = np.array([])
        if useloadtxt:
            for c,i in zip(coldefs,range(len(coldefs))):
                c = c.replace('\n','')
                if c:
                    if not delimiter or delimiter == ' ':
                        try:
                            if fitresheader:
                                self.__dict__[c] = np.loadtxt(filename,unpack=True,usecols=[i+1],skiprows=skiprows)
                            else:
                                self.__dict__[c] = np.loadtxt(filename,unpack=True,usecols=[i],skiprows=skiprows)
                        except:
                            if fitresheader:
                                self.__dict__[c] = np.loadtxt(filename,unpack=True,usecols=[i+1],dtype='string',skiprows=skiprows)
                            else:
                                self.__dict__[c] = np.loadtxt(filename,unpack=True,usecols=[i],dtype='string',skiprows=skiprows)
                    else:
                        try:
                            if fitresheader:
                                self.__dict__[c] = np.loadtxt(filename,unpack=True,usecols=[i+1],delimiter=',',skiprows=skiprows)
                            else:
                                self.__dict__[c] = np.loadtxt(filename,unpack=True,usecols=[i],delimiter=',',skiprows=skiprows)
                        except:
                            if fitresheader:
                                self.__dict__[c] = np.loadtxt(filename,unpack=True,usecols=[i+1],dtype='string',delimiter=',',skiprows=skiprows)
                            else:
                                self.__dict__[c] = np.loadtxt(filename,unpack=True,usecols=[i],dtype='string',delimiter=',',skiprows=skiprows)
                try:
                    self.filename = np.array([filename]*len(self.__dict__[c]))
                except:
                    for k in self.__dict__.keys(): self.__dict__[k] = np.array([self.__dict__[k]])
                    self.filename = np.array([filename]*len(self.__dict__[c]))
        else:
            fin = open(filename,'r')
            count = 0
            for line in fin:
                if count >= 1 and not line.startswith('#'):
                    entries = filter(None,line.split(' '))
                    for e,c in zip(entries,coldefs):
                        e = e.replace('\n','')
                        c = c.replace('\n','')
                        if not allstring:
                            try:
                                self.__dict__[c] = np.append(self.__dict__[c],float(e))
                            except:
                                self.__dict__[c] = np.append(self.__dict__[c],e)
                        else:
                            self.__dict__[c] = np.append(self.__dict__[c],e)
                        self.filename = np.append(self.filename,filename)
                else: count += 1
            fin.close()

    def addcol(self,col,data):
        self.__dict__[col] = data
    def cut_inrange(self,col,minval,maxval,rows=[]):
        if not len(rows):
            rows = np.where((self.__dict__[col] > minval) &
                            (self.__dict__[col] < maxval))[0]
            return(rows)
        else:
            rows2 = np.where((self.__dict__[col][rows] > minval) &
                            (self.__dict__[col][rows] < maxval))[0]
            return(rows[rows2])
    def appendfile(self,filename,useloadtxt=False):
        if useloadtxt:
            fin = open(filename,'r')
            for line in fin:
                if line.startswith('#'):
                    coldefs = filter(None,line.split('#')[1].split('\n')[0].split(' '))
                    break
            fin.close()
            for c,i in zip(coldefs,range(len(coldefs))):
                try:
                    self.__dict__[c] = np.concatenate((self.__dict__[c],np.loadtxt(filename,unpack=True,usecols=[i])))
                except:
                    self.__dict__[c] = np.concatenate((self.__dict__[c],np.loadtxt(filename,unpack=True,
                                                                                   usecols=[i],dtype='string')))
            self.filename = np.append(self.filename,np.array([filename]*len(np.loadtxt(filename,unpack=True,usecols=[i],dtype='string'))))
            
            return()
        fin = open(filename,'r')
        for line in fin:
            if line.startswith('#'):
                coldefs = filter(None,line.split('#')[1].split('\n')[0].split(' '))
            else:
                entries = filter(None,line.split(' '))
                for e,c in zip(entries,coldefs):
                    e = e.replace('\n','')
                    c = c.replace('\n','')
                    try:
                        self.__dict__[c] = np.append(self.__dict__[c],float(e))
                    except:
                        self.__dict__[c] = np.append(self.__dict__[c],e)
                self.filename = np.append(self.filename,filename)
