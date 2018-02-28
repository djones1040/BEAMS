#!/usr/bin/env python
# D. Jones - 5/14/14

import glob
import os
import numpy as np
#import pyfits

class txtobj:
    def __init__(self,filename,allstring=False,
                 cmpheader=False,sexheader=False,
                 useloadtxt=True,fitresheader=False,
                 delimiter=' ',skiprows=0,tabsep=False):
        if cmpheader: hdr = pyfits.getheader(filename)

        coldefs = np.array([])
        if cmpheader:
            for k,v in zip(list(hdr.keys()),list(hdr.values())):
                if 'COLTBL' in k and k != 'NCOLTBL':
                    coldefs = np.append(coldefs,v)
        elif sexheader:
            fin = open(filename,'r')
            lines = fin.readlines()
            for l in lines:
                if l.startswith('#'):
                    coldefs = np.append(coldefs,[_f for _f in l.split(' ') if _f][2])
        elif fitresheader:
            skiprows=0
            fin = open(filename,'r')
            lines = fin.readlines()
            for l in lines:
                skiprows += 1
                if l.startswith('VARNAMES:'):
                    l = l.replace('\n','')
                    coldefs = np.array([_f for _f in l.split(' ') if _f])
                    coldefs = coldefs[np.where((coldefs != 'VARNAMES:') & (coldefs != '\n') & (coldefs != '#'))]
                    break
                elif l.startswith('SN: '): break

        else:
            fin = open(filename,'r')
            lines = fin.readlines()
            if not tabsep:
                coldefs = np.array([_f for _f in lines[0].split(delimiter) if _f])
                coldefs = coldefs[np.where(coldefs != '#')]
            else:
                l = lines[0].replace('\n','')
                coldefs = np.array([_f for _f in l.split('\t') if _f])
                coldefs = coldefs[np.where(coldefs != '#')]
        for i in range(len(coldefs)):
            coldefs[i] = coldefs[i].replace('\n','').replace('\t','').replace(' ','')
            if coldefs[i]:
                self.__dict__[coldefs[i]] = np.array([])

        self.filename = np.array([])
        if useloadtxt:
            for c,i in zip(coldefs,list(range(len(coldefs)))):
                c = c.replace('\n','')
                if c:
                    if not delimiter or delimiter == ' ':
                        if fitresheader:
                            self.__dict__[c] = np.genfromtxt(filename,unpack=True,usecols=[i+1],skip_header=skiprows)
                        else:
                            self.__dict__[c] = np.genfromtxt(filename,unpack=True,usecols=[i],skip_header=skiprows)
                        if not len(self.__dict__[c][self.__dict__[c] == self.__dict__[c]]):
                            if fitresheader:
                                self.__dict__[c] = np.genfromtxt(filename,unpack=True,usecols=[i+1],dtype=str,skip_header=skiprows)
                            else:
                                self.__dict__[c] = np.genfromtxt(filename,unpack=True,usecols=[i],dtype=str,skip_header=skiprows)
                    else:
                        if fitresheader:
                            self.__dict__[c] = np.genfromtxt(filename,unpack=True,usecols=[i+1],delimiter=',',skip_header=skiprows)
                        else:
                            self.__dict__[c] = np.genfromtxt(filename,unpack=True,usecols=[i],delimiter=',',skip_header=skiprows)
                        if not len(self.__dict__[c][self.__dict__[c] == self.__dict__[c]]):
                            if fitresheader:
                                self.__dict__[c] = np.genfromtxt(filename,unpack=True,usecols=[i+1],dtype=str,delimiter=',',skip_header=skiprows)
                            else:
                                self.__dict__[c] = np.genfromtxt(filename,unpack=True,usecols=[i],dtype=str,delimiter=',',skip_header=skiprows)
                try:
                    self.filename = np.array([filename]*len(self.__dict__[c]))
                except:
                    for k in list(self.__dict__.keys()): self.__dict__[k] = np.array([self.__dict__[k]])
                    self.filename = np.array([filename]*len(self.__dict__[c]))
        else:
            fin = open(filename,'r')
            count = 0
            for line in fin:
                if count >= 1 and not line.startswith('#'):
                    entries = [_f for _f in line.split(' ') if _f]
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
                    coldefs = [_f for _f in line.split('#')[1].split('\n')[0].split(' ') if _f]
                    break
            fin.close()
            for c,i in zip(coldefs,list(range(len(coldefs)))):
                try:
                    self.__dict__[c] = np.concatenate((self.__dict__[c],np.genfromtxt(filename,unpack=True,usecols=[i])))
                except:
                    self.__dict__[c] = np.concatenate((self.__dict__[c],np.genfromtxt(filename,unpack=True,
                                                                                   usecols=[i],dtype=str)))
            self.filename = np.append(self.filename,np.array([filename]*len(np.genfromtxt(filename,unpack=True,usecols=[i],dtype=str))))
            
            return()
        fin = open(filename,'r')
        for line in fin:
            if line.startswith('#'):
                coldefs = [_f for _f in line.split('#')[1].split('\n')[0].split(' ') if _f]
            else:
                entries = [_f for _f in line.split(' ') if _f]
                for e,c in zip(entries,coldefs):
                    e = e.replace('\n','')
                    c = c.replace('\n','')
                    try:
                        self.__dict__[c] = np.append(self.__dict__[c],float(e))
                    except:
                        self.__dict__[c] = np.append(self.__dict__[c],e)
                self.filename = np.append(self.filename,filename)
