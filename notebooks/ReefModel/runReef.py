import os
import re
import shutil
import subprocess
from subprocess import call
from pathlib import Path

import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy import interpolate
from PIL import Image
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.colors import ListedColormap

def runReefModel(foutput='model'):

    dirpath = Path(foutput)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    # Check if the excecutable exist
    reef_code = Path("reef2D")
    if not reef_code.is_file():
        compilescript = 'gfortran ./code/REEF-series.f95 -fdefault-real-8 -O0 -fno-automatic  -o reef2D'
        process = subprocess.Popen(compilescript.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

    # Read parameters
    params = []
    with open('param0.dat') as f:
        for line in f:
            params.append(line.split()[0])

    sealvlfile = params[0]
    slp = float(params[1])/100.
    dt = float(params[-3])
    uprate = float(params[7])
    os.system('cp data/'+str(sealvlfile)+' .')

    # Run the model
    runscript = './reef2D'
    rc = call(runscript, shell=True)

    savepath = foutput
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    mvdata = 'mv box* rsl* vrel* profile* topo* volreef* '+str(savepath)+'/'
    os.system(mvdata)
    os.system('rm parame.dat')
    os.system('rm '+str(sealvlfile))
    os.system('cp param0.dat '+str(savepath))

    return

def readReefOutput(foutput):

    # Read parameters
    params = []
    with open(str(foutput)+'/param0.dat') as f:
        for line in f:
            params.append(line.split()[0])

    sealvlfile = params[0]
    slp = float(params[1])/100.
    dt = float(params[-3])
    uprate = float(params[7])

    topofile = glob.glob(str(foutput)+"/topo*")
    topo = pd.read_csv(topofile[0],delimiter='\s+', names=['x','y'], header=None)
    boxfile = glob.glob(str(foutput)+"/box*")
    box = pd.read_csv(boxfile[0],delimiter='\s+', header=None)
    rslfile = glob.glob(str(foutput)+"/rsl*")
    sea = pd.read_csv(rslfile[0],delimiter='\s+', names=['x','y'], header=None)
    if sealvlfile == 'waelbroeck2002.dat':
        sealvl = pd.read_csv('data/'+sealvlfile,delimiter='\s+', names=['time','sl','slmin','slmax'], header=None)
    else:
        sealvl = pd.read_csv('data/'+sealvlfile,delimiter='\s+', names=['time','sl'], header=None)
    volfile = glob.glob(str(foutput)+"/volreef*")
    vol = pd.read_csv(volfile[0], delimiter='\s+', names=['time','vol'], header=None)

    listoutfile = glob.glob(str(foutput)+"/profile*")

    nbb = len(foutput)+1
    mis = []
    filenb = []
    for k in range(len(listoutfile)):
        mis.append(int(listoutfile[k][nbb+7:nbb+9]))
        filenb.append(int(listoutfile[k][nbb+9:nbb+13]))
    data = {'file':listoutfile,
            'nb':filenb,
            'mis':mis
           }
    fout = pd.DataFrame(data)
    fout.sort_values('nb', ascending=False, ignore_index=True, inplace=True)

    # Initial basement
    dj = 1
    n = len(sealvl)-1
    uplift = np.zeros(n)
    for i in range(1, n):
        uplift[i]=uplift[i-1]+uprate*dt/1000
    upmax=uplift[-1]
    emin = sealvl.sl.min()
    emax = sealvl.sl.max()
    slopi=np.arctan(slp)
    jmax=int((np.abs(emin-emax)+np.abs(upmax)+3500)/(np.cos(np.pi/2.-slopi)))
    if uprate<=0:
        shifty=abs(emin-emax)+100
    else:
        shifty=abs(emin-emax)+upmax+100

    basex = np.zeros(jmax)
    basey = np.zeros(jmax)

    basey[0]=basex[0]*np.tan(slopi)-shifty
    for k in range(1,jmax):
        basex[k] = basex[k-1]+dj*np.cos(slopi)
        basey[k] = basex[k]*np.tan(slopi)-shifty

    reefLay = []
    for k in range(len(fout)):
        fname = fout['file'].iloc[k]
        out = pd.read_csv(fname,delimiter='\s+', names=['x','y'], header=None)
        reefLay.append(out)

    dist = np.arange(box.iloc[0][0], box.iloc[0][1])

    return [fout, reefLay, dist, basex, basey, sealvl, uprate, box, vol]

def plotFigNoSL(stp, outdata):

    fout = outdata[0]
    reefLay = outdata[1]
    dist = outdata[2]
    basex = outdata[3]
    basey = outdata[4]
    sealvl = outdata[5]
    uprate = outdata[6]
    box = outdata[7]

    timevals = np.zeros(len(fout)+1)
    timevals[0] = -(fout['nb'].iloc[0]+1)
    for k in range(1,len(timevals)):
        timevals[k] = timevals[k-1]+1

    fig, ax1 = plt.subplots(1,1,figsize=(8,5))

    cmap = plt.get_cmap('PiYG', fout['mis'].max())

    # Time
    time = fout['nb'].iloc[stp-1]

    # Top profile
    out = reefLay[stp-1]
    topx = out['x'].values
    topy = out['y'].values
    f = interpolate.interp1d(topx, topy, kind='linear')
    topy = f(dist)
    ax1.plot(dist, topy, '-', color='k', zorder=3, lw=2)


    # Sealevel
    ax1.plot([-basex.min(),-basex.max()],[sealvl['sl'][stp],sealvl['sl'][stp]],
             '-', color='b', zorder=1, lw=0.5)
    ax1.fill_between([-basex.min(),-basex.max()],-1000,[sealvl['sl'][stp],sealvl['sl'][stp]],
                     color='tab:blue', zorder=1, alpha=0.2)


    # MIS layers
    MIS = []
    MIScol = []
    mislay = fout['mis'][stp-1]
    MIS.append(topy)
    MIScol.append(mislay)

    for k in range(stp-2,-1,-1):

        out = reefLay[k]
        newx = out['x'].values
        newy = out['y'].values+uprate*(stp-k)

        f = interpolate.interp1d(newx, newy, kind='linear')
        tmpy = f(dist)
        topy = np.minimum(topy,tmpy)
        ax1.plot(dist, topy, '-', color='k', zorder=3, lw=0.2)

        if mislay != fout['mis'][k]:
            MIS.append(topy)
            mislay = fout['mis'][k]
            MIScol.append(mislay)

    for k in range(1,len(MIS)):
        ax1.plot(dist, MIS[k], '-', color='k', zorder=3, lw=1.5)
        ax1.fill_between(dist,MIS[k],MIS[k-1], color=cmap(len(MIS)-k), zorder=1)

    # Basement
    f = interpolate.interp1d(-basex, basey+uprate*stp, kind='linear')
    bed = f(dist)
    ax1.plot(dist, bed, '-', color='k', zorder=3, lw=2)
    ax1.fill_between(dist,bed,MIS[len(MIS)-1], color=cmap(0), zorder=1)
    ax1.fill_between(dist,-1000,bed, color='lightgray', zorder=1)


    ax1.set_xlim([box.iloc[0][0],box.iloc[0][1]])
    ax1.set_ylim([-300,box.iloc[0][3]])

    ax1.set_ylabel('Elevation (m)', fontsize=9, weight='bold')
    ax1.set_xlabel('Distance (m)', fontsize=9, weight='bold')
    ax1.text(box.iloc[0][1]-1200, box.iloc[0][3]-20, str(time)+' kyr', weight='bold', fontsize = 10, color = 'k')
    ax1.tick_params(axis="x", labelsize=8)
    ax1.tick_params(axis="y", labelsize=8)

    divider = make_axes_locatable(ax1)
    cax2 = divider.append_axes("right", size="1.5%", pad=0.1)

    cb2 = mpl.colorbar.ColorbarBase(cax2, cmap=cmap, ticks=[0, 1, 0.1])


    cb2 = mpl.colorbar.ColorbarBase(cax2, cmap=cmap,
                                    ticks=np.arange(0,1.01,1/fout['mis'].max()))
    cb2.ax.set_yticklabels(np.arange(len(np.arange(0,1.01,1/fout['mis'].max()))-1,-1,-1),
                           fontsize=6, weight='bold')
    cb2.set_label('Marine Isotope Stage',fontsize=8, weight='bold') #, rotation=270)
    plt.show()

def plotReef(outdata,zrange):

    fout = outdata[0]
    reefLay = outdata[1]
    dist = outdata[2]
    basex = outdata[3]
    basey = outdata[4]
    sealvl = outdata[5]
    uprate = outdata[6]
    box = outdata[7]

    timevals = np.zeros(len(fout)+1)
    timevals[0] = -(fout['nb'].iloc[0]+1)
    for k in range(1,len(timevals)):
        timevals[k] = timevals[k-1]+1

    step = len(fout)
    profiles = []
    for k in range(step):
        out = reefLay[k]
        topx = out['x'].values
        topy = out['y'].values
        f = interpolate.interp1d(topx, topy, kind='linear')
        pz = f(dist)
        profiles.append(pz)

    for k in range(step):
        profiles[k] += uprate*(step-k)

    lay = []
    lay.append(profiles[-1])
    p = 0
    for k in range(step-2,-1,-1):
        lay.append(np.minimum(lay[p],profiles[k]))
        p += 1

    ztop = lay[0].max()
    zbot = lay[0].min()
    f = interpolate.interp1d([dist[0],dist[-1]], [ztop,zbot], kind='linear')
    zbase = f(dist)
    zbase = np.minimum(zbase,lay[-1])
    lay.append(zbase)

    cmap = plt.get_cmap('RdPu_r', len(lay))

    rdpu = cm.get_cmap('RdPu', len(lay))
    newcolors = rdpu(np.linspace(0, 1, len(lay)))
    grey = np.array([189/256, 189/256, 189/256, 1])
    newcolors[:1, :] = grey
    newcmp = ListedColormap(newcolors)
    cmap2 = plt.get_cmap(newcmp, len(lay))

    fig, ax1 = plt.subplots(1,1,figsize=(10,5))
    for k in range(len(lay)):
        ax1.plot(dist, lay[k], '-', color='k', zorder=3, lw=0.8)
        ax1.fill_between(dist, lay[k],lay[k-1], color=cmap(k), alpha=0.8, zorder=1)

    ax1.plot(dist, lay[0], '-', color='k', zorder=3, lw=1)
    ax1.plot(dist, lay[-1], '-', color='k', zorder=3, lw=1)

    # Sealevel
    ax1.plot([dist[0],dist[-1]],[sealvl['sl'][step],sealvl['sl'][step]],
             '-', color='navy', zorder=0, lw=0.8)
    ax1.fill_between([dist[0],dist[-1]],-1000,[sealvl['sl'][step],sealvl['sl'][step]],
                     color='tab:blue', zorder=0, alpha=0.2)

    ax1.fill_between(dist,-1000, lay[-1], color='lightgray', zorder=1)
    ax1.set_xlim([dist[0],dist[-1]])
    ax1.set_ylim([zrange[0],zrange[1]])
    ax1.set_ylabel('Elevation (m)', fontsize=9, weight='bold')
    ax1.set_xlabel('Distance (m)', fontsize=9, weight='bold')
    ax1.tick_params(axis="x", labelsize=8)
    ax1.tick_params(axis="y", labelsize=8)

    divider = make_axes_locatable(ax1)
    cax2 = divider.append_axes("right", size="1.5%", pad=0.1)
    ticks=np.arange(0,1.001,1/len(timevals[:step+1]))[1:]+0.5/len(timevals[:step+1])
    cb2 = mpl.colorbar.ColorbarBase(cax2, cmap=cmap2, ticks=ticks)
    cb2.ax.set_yticklabels(np.abs(timevals[:step+1]).astype(int),fontsize=6, weight='bold')
    cb2.set_label('Stratal layer interval (kyr)',fontsize=8, weight='bold') #, rotation=270)

    plt.show()

    return

def plotFigQuaternary(stp, fname, outdata, ylim=[-300,50], textpos=[-4000,25], view=False):

    fout = outdata[0]
    reefLay = outdata[1]
    dist = outdata[2]
    basex = outdata[3]
    basey = outdata[4]
    sealvl = outdata[5]
    uprate = outdata[6]
    box = outdata[7]

    timevals = np.zeros(len(fout)+1)
    timevals[0] = -(fout['nb'].iloc[0]+1)
    for k in range(1,len(timevals)):
        timevals[k] = timevals[k-1]+1

    fig, axs = plt.subplots(2,1,figsize=(8,6), gridspec_kw={'height_ratios': [4, 1]})
    ax1 = axs[0]
    ax2 = axs[1]

    cmap = plt.get_cmap('Spectral', 101)

    # Time
    time = fout['nb'].iloc[stp-1]

    # Top profile
    out = reefLay[stp-1]
    topx = out['x'].values
    topy = out['y'].values
    f = interpolate.interp1d(topx, topy, kind='linear')
    topy = f(dist)
    ax1.plot(dist, topy, '-', color='k', zorder=3, lw=2)


    # Sealevel
    ax1.plot([-basex.min(),-basex.max()],[sealvl['sl'][stp],sealvl['sl'][stp]],
             '-', color='b', zorder=1, lw=0.5)
    ax1.fill_between([-basex.min(),-basex.max()],-1000,[sealvl['sl'][stp],sealvl['sl'][stp]],
                     color='tab:blue', zorder=1, alpha=0.2)

    ttop = topy.copy()
    d = 1
    for k in range(stp-2,-1,-1):

        out = reefLay[k]
        newx = out['x'].values
        newy = out['y'].values+uprate*(stp-k)

        f = interpolate.interp1d(newx, newy, kind='linear')
        tmpy = f(dist)
        topy = np.minimum(topy,tmpy)
        if k%10 ==0:
            ax1.plot(dist, topy, '-', color='k', zorder=3, lw=1.5)
            ax1.fill_between(dist,topy,ttop, color=cmap(d), zorder=1)
            d += 1
            ttop = topy.copy()
        else:
            ax1.plot(dist, topy, '-', color='k', zorder=3, lw=0.2)

    # Basement
    f = interpolate.interp1d(-basex, basey+uprate*stp, kind='linear')
    bed = f(dist)
    ax1.plot(dist, bed, '-', color='k', zorder=3, lw=2)
    ax1.fill_between(dist,bed,ttop, color=cmap(d), zorder=1)
    ax1.fill_between(dist,-1000,bed, color='lightgray', zorder=1)


    ax1.set_xlim([box.iloc[0][0],box.iloc[0][1]])
    ax1.set_ylim([ylim[0], ylim[1]])

    ax1.set_ylabel('Elevation (m)', fontsize=9, weight='bold')
    ax1.set_xlabel('Distance (m)', fontsize=9, weight='bold')
    ax1.text(textpos[0], textpos[1], str(time)+' kyr', weight='bold', fontsize = 10, color = 'k')
    ax1.tick_params(axis="x", labelsize=8)
    ax1.tick_params(axis="y", labelsize=8)


    sealvl['ntime'] = -sealvl['time']
    ax2.plot(sealvl['ntime'],sealvl['sl'],'-', color='k', zorder=2, lw=2)
    ax2.set_ylabel('Sea-level (m)', fontsize=9, weight='bold')
    ax2.set_xlabel('Time (kyr)', fontsize=9, weight='bold')

    ax2.set_xlim([sealvl['ntime'].min(),sealvl['ntime'].max()])
    # ax2.set_ylim([sealvl['slmax'].min()-40,sealvl['slmin'].max()+20])
    ax2.set_ylim([sealvl['sl'].min()-40,sealvl['sl'].max()+20])

    # minsl = sealvl['slmax'].min()-5
    minsl = sealvl['sl'].min()-5
    ax2.plot([sealvl['ntime'].min(),sealvl['ntime'].max()],[minsl,minsl],'-',
             color='k', zorder=3, lw=1)

    d = 0
    cmap2 = plt.get_cmap('Spectral_r', 101)
    for k in range(1,len(timevals)-1):
        colmap = cmap(fout['mis'].iloc[np.where(fout['nb'] == k)[0]].values[0])
        ax2.plot([timevals[k-1],timevals[k]],[minsl-100,minsl-6], color=cmap2(d), zorder=1, lw=4)
        if k%10 == 0:
            d += 1

    ax2.plot([-time,-time],[minsl,1000], color='b', zorder=1, lw=5, alpha=0.25)

    ax2.tick_params(axis="x", labelsize=8)
    ax2.tick_params(axis="y", labelsize=8)
    if view:
        plt.show()
    else:
        fig.savefig(fname+str(stp)+'.png',dpi=500,transparent=False)
        plt.close(fig)

    return

def plotEvolution(outdata):

    sealvl = outdata[5]
    vol = outdata[8]

    vpeaks, _ = find_peaks(vol.vol,height=50,distance=10)

    speaks, _ = find_peaks(sealvl['sl'],distance=10)
    peaks = np.zeros((len(speaks)+1),dtype=int)
    peaks[:-1] = speaks
    peaks[-1] = len(vol)-1
    speaks = peaks.copy()
    speaks = speaks[np.where(sealvl['sl'][speaks]>-50)[0]]

    flip_peaks = np.flip(speaks)
    flip_peakv = np.flip(vpeaks)

    d = 0
    nextp = []
    for k in range(len(flip_peaks)):
        slpeak = flip_peaks[k]
        cont = True
        while cont:
            if flip_peakv[d]<slpeak:
                nextp.append(flip_peakv[d])
                cont = False
            else:
                d += 1
                if d >= len(flip_peakv):
                    cont = False
                    nextp.append(slpeak-1)

    peakvol = np.flip(np.asarray(nextp))

    fig, ax1 = plt.subplots(1,1,figsize=(9,3))
    ax1.plot(vol['time'], vol['vol']/1000., '-', color='b', zorder=3, lw=1, alpha=0.5)
    ax1.fill_between(vol['time'],-100,vol['vol']/1000., color='b', zorder=2, alpha=0.5)
    plt.xlim(0,1000)
    plt.ylim(0,4)
    ax1.tick_params(axis="x", labelsize=8)
    ax1.tick_params(axis="y", labelsize=8, labelcolor='b')
    ax1.set_ylabel('productivity (m$^3$/yr)', fontsize=9, weight='bold',color='b')
    ax1.set_xlabel('time (kyr)', fontsize=9, weight='bold')

    ax2 = ax1.twinx()
    ax2.set_ylabel('rsl (m)', color='tab:red', fontsize=9, weight='bold')
    ax2.plot(sealvl['time'], sealvl['sl'], color='tab:red', lw=2)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim([-200,2])
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    offtime = []
    offset = []
    offsl = []
    offprod = []
    for k in range(len(speaks)):
        x = vol['time']
        x2 = vol['time'][peakvol[k]]
        x1 = vol['time'][speaks[k]]
        offsl.append(sealvl['sl'][peakvol[k]]-50.)
        offset.append(-(x2-x1))
        offtime.append(vol['time'][x2])
        offprod.append(vol['vol'][peakvol[k]]/1000.)
        ax1.fill_between(x,-100,10,where=(x>=x1) & (x<=x2), color='lightgray', zorder=1, alpha=0.5)

    plt.tight_layout()
    plt.show()

    fig, ax2 = plt.subplots(1,1,figsize=(7,3))
    ax2.scatter(offtime,offset,c=offsl,cmap='seismic',s=np.asarray(offprod)*5, edgecolor='black', linewidth=0.2)
    plt.xlim(0,1000)
    plt.ylim(-22,4)
    ax2.plot([0,1000],[0,0],lw=0.1,c='k')
    ax2.set_ylabel('max. prod. offset (kyr)', fontsize=9, weight='bold')
    ax2.set_xlabel('time (kyr)', fontsize=9, weight='bold')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()

    data = {'rate':np.diff(sealvl['sl']),
            'prod':vol['vol']/1000.
           }
    df = pd.DataFrame(data)
    df = df[df['prod'] > 0]
    df.sort_values('rate', ascending=False, ignore_index=True, inplace=True)

    df['rmean'] = df['prod'].rolling(75,center=True).mean()
    df['rmed'] = df['prod'].rolling(75,center=True).median()

    rmean = np.asarray(df['rmean'])
    rmean = rmean[~np.isnan(rmean)]

    dfnonan = df.dropna()
    rmean = np.asarray(dfnonan['rmean'])
    rmed = np.asarray(dfnonan['rmed'])
    slrate = np.asarray(dfnonan['rate'])

    rmeansmth = savgol_filter(rmean, 81, 3)
    rmedsmth = savgol_filter(rmed, 81, 3)

    fig, ax2 = plt.subplots(1,1,figsize=(6,6))
    ax2.set_xlabel('rsl rate (mm/yr)', color='k', fontsize=9, weight='bold')
    ax2.set_ylabel('productivity (m$^3$/yr)', color='k', fontsize=9, weight='bold')
    ax2.scatter(df['rate'],df['prod'],s=5,c='k')
    ax2.plot(df['rate'],df['rmean'],lw=2, c='r',alpha=0.5)
    ax2.plot(df['rate'],df['rmed'],lw=2,c='lime',alpha=0.5)

    ax2.plot(slrate, rmeansmth, lw=3, c='r',label='mean')
    ax2.plot(slrate,rmedsmth, lw=3, c='lime',label='median')

    ax2.tick_params(axis='y', labelcolor='k')
    ax2.set_ylim([-0.01,2.5])
    ax2.set_xlim([-7,12])
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.legend(frameon=False, prop={'size': 9}, loc=2)
    plt.show()

    return

def plotFig(stp, fname, outdata, ylim=[-300,50], textpos=[-4000,25], view=False):

    fout = outdata[0]
    reefLay = outdata[1]
    dist = outdata[2]
    basex = outdata[3]
    basey = outdata[4]
    sealvl = outdata[5]
    uprate = outdata[6]
    box = outdata[7]

    timevals = np.zeros(len(fout)+1)
    timevals[0] = -(fout['nb'].iloc[0]+1)
    for k in range(1,len(timevals)):
        timevals[k] = timevals[k-1]+1

    fig, axs = plt.subplots(2,1,figsize=(8,7), gridspec_kw={'height_ratios': [4, 1]})
    ax1 = axs[0]
    ax2 = axs[1]

    cmap = plt.get_cmap('PiYG', fout['mis'].max())

    # Time
    time = fout['nb'].iloc[stp-1]

    # Top profile
    out = reefLay[stp-1]
    topx = out['x'].values
    topy = out['y'].values
    f = interpolate.interp1d(topx, topy, kind='linear')
    topy = f(dist)
    ax1.plot(dist, topy, '-', color='k', zorder=3, lw=2)


    # Sealevel
    ax1.plot([-basex.min(),-basex.max()],[sealvl['sl'][stp],sealvl['sl'][stp]],
             '-', color='b', zorder=1, lw=0.5)
    ax1.fill_between([-basex.min(),-basex.max()],-1000,[sealvl['sl'][stp],sealvl['sl'][stp]],
                     color='tab:blue', zorder=1, alpha=0.2)

    # MIS layers
    MIS = []
    MIScol = []
    mislay = fout['mis'][stp-1]
    MIS.append(topy)
    MIScol.append(mislay)

    for k in range(stp-2,-1,-1):

        out = reefLay[k]
        newx = out['x'].values
        newy = out['y'].values+uprate*(stp-k)

        f = interpolate.interp1d(newx, newy, kind='linear')
        tmpy = f(dist)
        topy = np.minimum(topy,tmpy)
        ax1.plot(dist, topy, '-', color='k', zorder=3, lw=0.2)

        if mislay != fout['mis'][k]:
            MIS.append(topy)
            mislay = fout['mis'][k]
            MIScol.append(mislay)

    for k in range(1,len(MIS)):
        ax1.plot(dist, MIS[k], '-', color='k', zorder=3, lw=1.5)
        ax1.fill_between(dist,MIS[k],MIS[k-1], color=cmap(len(MIS)-k), zorder=1)

    # Basement
    f = interpolate.interp1d(-basex, basey+uprate*stp, kind='linear')
    bed = f(dist)
    ax1.plot(dist, bed, '-', color='k', zorder=3, lw=2)
    ax1.fill_between(dist,bed,MIS[len(MIS)-1], color=cmap(0), zorder=1)
    ax1.fill_between(dist,-1000,bed, color='lightgray', zorder=1)


    ax1.set_xlim([box.iloc[0][0],box.iloc[0][1]])
    ax1.set_ylim([ylim[0], ylim[1]])

    ax1.set_ylabel('Elevation (m)', fontsize=9, weight='bold')
    ax1.set_xlabel('Distance (m)', fontsize=9, weight='bold')
    ax1.text(textpos[0], textpos[1], str(time)+' kyr', weight='bold', fontsize = 10, color = 'k')
    ax1.tick_params(axis="x", labelsize=8)
    ax1.tick_params(axis="y", labelsize=8)

    divider = make_axes_locatable(ax1)
    cax2 = divider.append_axes("right", size="1.5%", pad=0.1)

    cb2 = mpl.colorbar.ColorbarBase(cax2, cmap=cmap, ticks=[0, 1, 0.1])


    cb2 = mpl.colorbar.ColorbarBase(cax2, cmap=cmap,
                                    ticks=np.arange(0,1.01,1/fout['mis'].max()))
    cb2.ax.set_yticklabels(np.arange(len(np.arange(0,1.01,1/fout['mis'].max()))-1,-1,-1),
                           fontsize=6, weight='bold')
    cb2.set_label('Marine Isotope Stage',fontsize=8, weight='bold')


    sealvl['ntime'] = -sealvl['time']
    ax2.plot(sealvl['ntime'],sealvl['sl'],'-', color='k', zorder=2, lw=2)
    # ax2.fill_between(sealvl['ntime'],sealvl['slmin'],sealvl['slmax'], color='lightgray', zorder=1)
    ax2.set_ylabel('Sea-level (m)', fontsize=9, weight='bold')
    ax2.set_xlabel('Time (kyr)', fontsize=9, weight='bold')

    ax2.set_xlim([sealvl['ntime'].min(),sealvl['ntime'].max()])
    # ax2.set_ylim([sealvl['slmax'].min()-40,sealvl['slmin'].max()+20])
    ax2.set_ylim([sealvl['sl'].min()-40,sealvl['sl'].max()+20])

    # minsl = sealvl['slmax'].min()-5
    minsl = sealvl['sl'].min()-5
    ax2.plot([sealvl['ntime'].min(),sealvl['ntime'].max()],[minsl,minsl],'-',
             color='k', zorder=3, lw=1)

    for k in range(1,len(timevals)-1):
        colmap = cmap(fout['mis'].iloc[np.where(fout['nb'] == k)[0]].values[0])
        ax2.plot([timevals[k-1],timevals[k]],[minsl-100,minsl-6], color=colmap, zorder=1, lw=4)

    ax2.plot([-time,-time],[minsl,1000], color='b', zorder=1, lw=5, alpha=0.25)

    ax2.tick_params(axis="x", labelsize=8)
    ax2.tick_params(axis="y", labelsize=8)
    if view:
        plt.show()
    else:
        fig.savefig(fname+str(stp)+'.png',dpi=500,transparent=False)
        plt.close(fig)

    return

def makeMovie(fname, outdata, step=5, ylim=[-300,50], textpos=[-4000,25]):

    imgs = []
    # For running over the entire time period remove the next line
    # and uncomment the following one
    for k in range(1,431,step):
        plotFig(k, fname, outdata, ylim, textpos, view=False) #plotfig(k,'model1')
        imgs.append(fname+str(k)+".png")


    # Create the frames
    frames = []
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(fname+'.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=500, loop=0)
