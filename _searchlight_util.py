
import numpy as np
import os
import deepdish as dd
import scipy.stats as stats 
from scipy.stats import norm #for survival fucntion

nv = 40962

def SLtoVox(D, SLlist, nv, zeronan=True):
    
    '''
    # D is dict of L, R, with N x arbitrary dims
    # SLlist is dict of L, R list of length N, with vertices for each SL
    '''


    Dvox = {}
    Dcount = {}
    for hem in ['L', 'R']:
        Dvox[hem] = np.zeros((nv,)+ D[hem].shape[1:])
        Dcount[hem] = np.zeros((nv,)+(1,)*len(D[hem].shape[1:]))
        for i in range(len(SLlist[hem])):
        #for i in range(530):
            Dvox[hem][SLlist[hem][i]] += D[hem][i]
            Dcount[hem][SLlist[hem][i]] += 1

        Dcount[hem][Dcount[hem] == 0] = np.nan
        Dvox[hem] = Dvox[hem] / Dcount[hem]

        if zeronan:
            Dvox[hem][np.isnan(Dvox[hem])] = 0

    return Dvox

def nullZ(X):
    '''
    # Last dimension of X is nPerm+1, with real data at 0 element
    '''

    X_roll = np.rollaxis(X, len(X.shape)-1)
    
    means = np.nanmean(X_roll[1:],0)
#     means = X_roll[1:].mean(axis=0)
    std = np.nanstd(X_roll[1:],0)
#     std = X_roll[1:].std(axis=0)
    if len(X.shape) > 1:
        std[std==0] = np.nan
    Z = (X_roll[0] - means) / std
    return Z

# Z is dict with L,R, nan indicates invalid verts
# Returns q values for each hem
def FDR_z_hem(Z,sided=2):
    '''
    # Z is dict with L,R, nan indicates invalid verts
    # Returns q values for each hem
    # ---> z_cat = np.concatenate((Z['L'], Z['R']))
    '''
    z_cat = np.concatenate((Z['L'], Z['R']))
    valid_inds = np.logical_not(np.isnan(z_cat))
    q_cat = np.ones(z_cat.shape[0])
    
    ## 20220912 modification
    if sided==1:
        q_cat[valid_inds] = FDR_p(stats.norm.sf(z_cat[valid_inds]))
    elif sided==2:
        p = stats.norm.sf(np.abs(z_cat[valid_inds]))*2
        q_cat[valid_inds] = FDR_p(p)
        
    q = {}
    q['L'] = q_cat[:Z['L'].shape[0]]
    q['R'] = q_cat[Z['R'].shape[0]:]

    return q

def FDR_p(pvals):
    '''
    # Port of AFNI mri_fdrize.c
    
    one hem at a time?
    
    '''
    assert np.all(pvals>=0) and np.all(pvals<=1)
    pvals[pvals < np.finfo(np.float_).eps] = np.finfo(np.float_).eps
    pvals[pvals == 1] = 1-np.finfo(np.float_).eps
    n = pvals.shape[0]

    qvals = np.zeros((n))
    sorted_ind = np.argsort(pvals)
    sorted_pvals = pvals[sorted_ind]
    qmin = 1.0
    for i in range(n-1,-1,-1):
        qval = (n * sorted_pvals[i])/(i+1)
        if qval > qmin:
            qval = qmin
        else:
            qmin = qval
        qvals[sorted_ind[i]] = qval

    # Estimate number of true positives m1 and adjust q
    if n >= 233:
        phist = np.histogram(pvals, bins=20, range=(0, 1))[0]
        sorted_phist = np.sort(phist[3:19])
        if np.sum(sorted_phist) >= 160:
            median4 = n - 20*np.dot(np.array([1, 2, 2, 1]), sorted_phist[6:10])/6
            median6 = n - 20*np.dot(np.array([1, 2, 2, 2, 2, 1]), sorted_phist[5:11])/10
            m1 = min(median4, median6)

            qfac = (n - m1)/n
            if qfac < 0.5:
                qfac = 0.25 + qfac**2
            qvals *= qfac

    return qvals

def FDR_p_hem(p):
    '''involves... p_cat = np.concatenate((p['L'], p['R']))'''
    p_cat = np.concatenate((p['L'], p['R']))
    valid_inds = np.logical_not(np.isnan(p_cat))
    q_cat = np.ones(p_cat.shape[0])
    q_cat[valid_inds] = FDR_p(p_cat[valid_inds])

    q = {}
    q['L'] = q_cat[:p['L'].shape[0]]
    q['R'] = q_cat[p['L'].shape[0]:]

    return q


def NonparametricP(dd_vox,sided=1):
    '''
    one or two-tailed non-parametric p-value calculation
    
    dd_vox is one of the hems, where last axis is permutation
    
    '''
    p_brain = np.zeros((nv))
    nPerm = dd_vox.shape[1] - 1 #number of permtutations
    ### find proportion of vertices greater than value of current vertex
    for v in range(nv): 
        if ~np.isnan(dd_vox[v,0]): 
            
            thesum = np.sum(np.abs(dd_vox[v,:])>=np.abs(dd_vox[v,0])) if sided==2 else np.sum((dd_vox[v,:])>=(dd_vox[v,0]))
                

            p_brain[v] = (thesum/(nPerm+1))  #turn to fraction
        else:
            p_brain[v] = np.nan
    return p_brain


# roi_path = '/jukebox/norman/rmasis/clones/SchemaBigFiles/draft_PAPER/ROIs'
roi_path = '../../PythonData2023/ROIs'
SLlist=dd.io.load(os.path.join(roi_path,'SLlist_c10.h5'))
ROIlist=dd.io.load(os.path.join(roi_path,'SLlist_c10.h5'))