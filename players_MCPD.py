import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.decomposition import PCA
from ruptures.detection import Pelt


pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)



class SSA(object):

  # code of the SSA class from the following link :  https://www.kaggle.com/code/jdarcy/introducing-ssa-for-time-series-decomposition 
    
    __supported_types = (pd.Series, np.ndarray, list)
    
    def __init__(self, tseries, L, save_mem=True):
        """
        Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
        recorded at equal intervals.
        
        Parameters
        ----------
        tseries : The original time series, in the form of a Pandas Series, NumPy array or list. 
        L : The window length. Must be an integer 2 <= L <= N/2, where N is the length of the time series.
        save_mem : Conserve memory by not retaining the elementary matrices. Recommended for long time series with
            thousands of values. Defaults to True.
        
        Note: Even if an NumPy array or list is used for the initial time series, all time series returned will be
        in the form of a Pandas Series or DataFrame object.
        """
        
        # Tedious type-checking for the initial time series
        if not isinstance(tseries, self.__supported_types):
            raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")
        
        # Checks to save us from ourselves
        self.N = len(tseries)
        if not 2 <= L <= self.N/2:
            raise ValueError("The window length must be in the interval [2, N/2].")
        
        self.L = L
        self.orig_TS = pd.Series(tseries)
        self.K = self.N - self.L + 1
        
        # Embed the time series in a trajectory matrix
        self.X = np.array([self.orig_TS.values[i:L+i] for i in range(0, self.K)]).T
        
        # Decompose the trajectory matrix
        self.U, self.Sigma, VT = np.linalg.svd(self.X)
        self.d = np.linalg.matrix_rank(self.X)
        
        self.TS_comps = np.zeros((self.N, self.d))
        
        if not save_mem:
            # Construct and save all the elementary matrices
            self.X_elem = np.array([ self.Sigma[i]*np.outer(self.U[:,i], VT[i,:]) for i in range(self.d) ])

            # Diagonally average the elementary matrices, store them as columns in array.           
            for i in range(self.d):
                X_rev = self.X_elem[i, ::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.V = VT.T
        else:
            # Reconstruct the elementary matrices without storing them
            for i in range(self.d):
                X_elem = self.Sigma[i]*np.outer(self.U[:,i], VT[i,:])
                X_rev = X_elem[::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.X_elem = "Re-run with save_mem=False to retain the elementary matrices."
            
            # The V array may also be very large under these circumstances, so we won't keep it.
            self.V = "Re-run with save_mem=False to retain the V matrix."
        
        # Calculate the w-correlation matrix.
        self.calc_wcorr()
            
    def components_to_df(self, n=0):
        """
        Returns all the time series components in a single Pandas DataFrame object.
        """
        if n > 0:
            n = min(n, self.d)
        else:
            n = self.d
        
        # Create list of columns - call them F0, F1, F2, ...
        cols = ["F{}".format(i) for i in range(n)]
        return pd.DataFrame(self.TS_comps[:, :n], columns=cols, index=self.orig_TS.index)
            
    
    def reconstruct(self, indices):
        """
        Reconstructs the time series from its elementary components, using the given indices. Returns a Pandas Series
        object with the reconstructed time series.
        
        Parameters
        ----------
        indices: An integer, list of integers or slice(n,m) object, representing the elementary components to sum.
        """
        if isinstance(indices, int): indices = [indices]
        
        ts_vals = self.TS_comps[:,indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)
    
    def calc_wcorr(self):
        """
        Calculates the w-correlation matrix for the time series.
        """
             
        # Calculate the weights
        w = np.array(list(np.arange(self.L)+1) + [self.L]*(self.K-self.L-1) + list(np.arange(self.L)+1)[::-1])
        
        def w_inner(F_i, F_j):
            return w.dot(F_i*F_j)
        
        # Calculated weighted norms, ||F_i||_w, then invert.
        F_wnorms = np.array([w_inner(self.TS_comps[:,i], self.TS_comps[:,i]) for i in range(self.d)])
        F_wnorms = F_wnorms**-0.5
        
        # Calculate Wcorr.
        self.Wcorr = np.identity(self.d)
        for i in range(self.d):
            for j in range(i+1,self.d):
                self.Wcorr[i,j] = abs(w_inner(self.TS_comps[:,i], self.TS_comps[:,j]) * F_wnorms[i] * F_wnorms[j])
                self.Wcorr[j,i] = self.Wcorr[i,j]
    


def apply_MCPD(position, beta=500000000, quantile_threshold_low=0.01, quantile_threshold_high=0.99, verbose=False):

    # compute PCA over the multivariate dataframe
    pca = PCA(n_components=0.99)
    eucl_pca = pca.fit_transform(position.values)
    ncomponents = pca.n_components_
    col_name = ['comp_{}'.format(i+1) for i in range(ncomponents)] 
    ugp_pca_df = pd.DataFrame(eucl_pca, columns=col_name, index=position.index)
    if verbose :
        print('NB COMP : ', ncomponents)
        print('EXPLAINED VAR : ')
        print(pca.explained_variance_)
        print('UGP PCA')
        print(ugp_pca_df.head())


    timelabel = np.arange(0, ugp_pca_df.index.size / 60, 1/60)

    if verbose :
        fig, ax = plt.subplots(2)
        ax[0].plot(timelabel, ugp_pca_df.comp_1.values)
        ax[1].plot(timelabel, ugp_pca_df.comp_2.values)
        plt.suptitle('Two first PCA components')
        plt.xlabel('Time [min]')
        plt.show()

    # Smooth values
    window = 100 # samples
    idx = 0
    while idx < ugp_pca_df.comp_2.values.size : 
        if ugp_pca_df.comp_2.values.shape[0] - idx < 5000 :
            pos_ssa = SSA(tseries=ugp_pca_df.comp_2.values[idx:], L=window)
        else :
            pos_ssa = SSA(tseries=ugp_pca_df.comp_2.values[idx:5000+idx], L=window)
        new_ssa = pos_ssa.components_to_df().loc[:, 'F0']
        if idx == 0:
            ugp_ssa_df = new_ssa 
        else :
            ugp_ssa_df = pd.concat([ugp_ssa_df, new_ssa], axis=0)
        idx += 5000
    ugp_ssa_df.index = ugp_pca_df.index

    if verbose :
        plt.figure(figsize=(25,5))
        plt.plot(timelabel, ugp_ssa_df.values, color='b')
        plt.title('SSA on 2nd PCA component (eucl)')
        plt.xlabel('Time [min]')
        plt.show()

    # Remove outliers : Histogram + threshold

    threshold_low, threshold_high = np.quantile(
        ugp_ssa_df.values, [quantile_threshold_low, quantile_threshold_high]
    )
    
    if verbose :
        fig, ax = plt.subplots(figsize=(25,5))
        _ = ax.hist(ugp_ssa_df.values, 20)
        _ = ax.axvline(threshold_low, ls="--", color="k")
        _ = ax.axvline(threshold_high, ls="--", color="k")
        plt.title('Histogram position : p-value = [{}, {}]'.format(quantile_threshold_low, quantile_threshold_high))
        plt.xlabel('Values')
        plt.show()
    
    outlier_mask = (ugp_ssa_df.values >= threshold_low) & (ugp_ssa_df.values <= threshold_high)
    ugp_ssa_wo_df = ugp_ssa_df[outlier_mask]
    timelabel = (ugp_ssa_wo_df.index - ugp_ssa_wo_df.index.values[0]).total_seconds() / 60
    
    if verbose :
        plt.figure(figsize=(25,5))
        plt.title('Denoised and filtered signal')
        plt.xlabel('Values')
        plt.plot(timelabel, ugp_ssa_wo_df.values)
        plt.show()

    # Compute PELT over the 3 first components
    pelt_cpd = Pelt()
    comp_2_pelt = pelt_cpd.fit_predict(ugp_ssa_wo_df.values, beta)
    
    chg_pts = []
    for i, chgp in enumerate(comp_2_pelt) :
        chg_pts.append(round(chgp/600, 1))
    
    dic = {}
    dic['time'] = comp_2_pelt
    dic['chg_pts'] = chg_pts
    chg_pts_df = pd.DataFrame(dic)
    
    if verbose :
        timelabel = (ugp_ssa_wo_df.index - ugp_ssa_wo_df.index.values[0]).total_seconds() / 60
        fig, ax = plt.subplots(figsize=(25,5))
        ax.plot(timelabel.values, ugp_ssa_wo_df.values)
        for chg_pt in chg_pts : 
            _ = ax.axvline(chg_pt, ls="--", color="r")
        plt.title('Detected change point in PCA component 2 : {}'.format(str(chg_pt)))
        plt.xlabel('Time [min]')
        plt.show()
    return chg_pts_df

