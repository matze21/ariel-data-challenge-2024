import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import glob
import os

import itertools
from astropy.stats import sigma_clip
import scipy.stats

from tqdm import tqdm
import time

def ADC_convert(signal, gain, offset):
    signal = signal.astype(np.float64)
    signal /= gain
    signal += offset
    return signal

def mask_hot_dead(signal, dead, dark):
    hot = sigma_clip(
        dark, sigma=5, maxiters=5
    ).mask
    hot = np.tile(hot, (signal.shape[0], 1, 1))
    dead = np.tile(dead, (signal.shape[0], 1, 1))
    signal = np.ma.masked_where(dead, signal)
    signal = np.ma.masked_where(hot, signal)
    return signal


def apply_linear_corr(linear_corr,clean_signal):
    linear_corr = np.flip(linear_corr, axis=0)
    for x, y in itertools.product(
                range(clean_signal.shape[1]), range(clean_signal.shape[2])
            ):
        poli = np.poly1d(linear_corr[:, x, y])
        clean_signal[:, x, y] = poli(clean_signal[:, x, y])
    return clean_signal

def clean_dark(signal, dead, dark, dt):

    dark = np.ma.masked_where(dead, dark)
    dark = np.tile(dark, (signal.shape[0], 1, 1))

    signal -= dark* dt[:, np.newaxis, np.newaxis]
    return signal

def get_cds(signal):
    """
    calcs difference in time becuase the detector is read twice, once in the beginning and once in the end of the measurement, the difference is what's measured
    """
    cds = signal[:,1::2,:,:] - signal[:,::2,:,:]
    return cds

def get_cds_origial(signal):
    """
    calcs difference in time becuase the detector is read twice, once in the beginning and once in the end of the measurement, the difference is what's measured
    """
    cds = signal[1::2,:,:] - signal[::2,:,:]
    return cds

def bin_obs(cds_signal,binning):
    cds_transposed = cds_signal.transpose(0,1,3,2)
    cds_binned = np.zeros((cds_transposed.shape[0], cds_transposed.shape[1]//binning, cds_transposed.shape[2], cds_transposed.shape[3]))
    for i in range(cds_transposed.shape[1]//binning):
        cds_binned[:,i,:,:] = np.sum(cds_transposed[:,i*binning:(i+1)*binning,:,:], axis=1)
    return cds_binned

def correct_flat_field(flat,dead, signal):
    flat = flat.transpose(1, 0)
    dead = dead.transpose(1, 0)
    flat = np.ma.masked_where(dead, flat)
    flat = np.tile(flat, (signal.shape[0], 1, 1))
    signal = signal / flat
    return signal


def calibrateData(planet_id,train_adc_info,axis_info,DO_MASK,DO_THE_NL_CORR,DO_DARK,DO_FLAT,TIME_BINNING):
    """
    DO_MASK         # filter out non responsive pixels
    DO_THE_NL_CORR  # nonlinear correction due to artefacts when reading pixels
    DO_DARK         # dark current is accumulating over time in the pixels, need to compensate that (seems like integration artefact)
    DO_FLAT         # pixel to pixel variation correction (e.g. how pixels respond differently when illuminated uniformly)
    TIME_BINNING    #do a time binning on choosen frequency
    DIFF_REP        # represent integral of one timestamp rather than the absolute reading
    """
    path_folder = ""  

    t0 = time.time()
    cut_inf, cut_sup = 39, 321
    l = cut_sup - cut_inf
    AIRS_CH0_clean = np.zeros((1, 11250, 32, l))
    FGS1_clean = np.zeros((1, 135000, 12, 12))
    df = pd.read_parquet(os.path.join(path_folder,f'train/{planet_id}/AIRS-CH0_signal.parquet'))
    signal = df.values.astype(np.float64).reshape((df.shape[0], 32, 356))
    gain = train_adc_info['AIRS-CH0_adc_gain'].loc[planet_id]
    offset = train_adc_info['AIRS-CH0_adc_offset'].loc[planet_id]
    signal = ADC_convert(signal, gain, offset)
    dt_airs = axis_info['AIRS-CH0-integration_time'].dropna().values
    dt_airs[1::2] += 0.1
    chopped_signal = signal[:, :, cut_inf:cut_sup]
    airs_original = chopped_signal
    del signal, df
    # CLEANING THE DATA: AIRS
    flat = pd.read_parquet(os.path.join(path_folder,f'train/{planet_id}/AIRS-CH0_calibration/flat.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
    dark = pd.read_parquet(os.path.join(path_folder,f'train/{planet_id}/AIRS-CH0_calibration/dark.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
    dead_airs = pd.read_parquet(os.path.join(path_folder,f'train/{planet_id}/AIRS-CH0_calibration/dead.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
    linear_corr = pd.read_parquet(os.path.join(path_folder,f'train/{planet_id}/AIRS-CH0_calibration/linear_corr.parquet')).values.astype(np.float64).reshape((6, 32, 356))[:, :, cut_inf:cut_sup]
    t1 = time.time()
    if DO_MASK:
        chopped_signal = mask_hot_dead(chopped_signal, dead_airs, dark)
        AIRS_CH0_clean[0] = chopped_signal
    else:
        AIRS_CH0_clean[0] = chopped_signal
    t2 = time.time()
    if DO_THE_NL_CORR: 
        linear_corr_signal = apply_linear_corr(linear_corr,AIRS_CH0_clean[0])
        AIRS_CH0_clean[0] = linear_corr_signal
    del linear_corr
    t3 = time.time()
    if DO_DARK: 
        cleaned_signal = clean_dark(AIRS_CH0_clean[0], dead_airs, dark,dt_airs)
        AIRS_CH0_clean[0] = cleaned_signal
    else: 
        pass
    del dark
    t4 = time.time()
    df = pd.read_parquet(os.path.join(path_folder,f'train/{planet_id}/FGS1_signal.parquet'))
    fgs_signal = df.values.astype(np.float64).reshape((df.shape[0], 32, 32))
    fgs_signal = fgs_signal[:,10:22,10:22]

    FGS1_gain = train_adc_info['FGS1_adc_gain'].loc[planet_id]
    FGS1_offset = train_adc_info['FGS1_adc_offset'].loc[planet_id]
    fgs_signal = ADC_convert(fgs_signal, FGS1_gain, FGS1_offset)
    dt_fgs1 = np.ones(len(fgs_signal))*0.1  ## please refer to data documentation for more information
    dt_fgs1[1::2] += 0.1
    chopped_FGS1 = fgs_signal
    fgs_original = chopped_FGS1
    del fgs_signal, df
    # CLEANING THE DATA: FGS1
    dark = pd.read_parquet(os.path.join(path_folder,f'train/{planet_id}/FGS1_calibration/dark.parquet')).values.astype(np.float64).reshape((32, 32))
    dark = dark[10:22,10:22]
    dead_fgs1 = pd.read_parquet(os.path.join(path_folder,f'train/{planet_id}/FGS1_calibration/dead.parquet')).values.astype(np.float64).reshape((32, 32))
    dead_fgs1 = dead_fgs1[10:22,10:22]
    linear_corr = pd.read_parquet(os.path.join(path_folder,f'train/{planet_id}/FGS1_calibration/linear_corr.parquet')).values.astype(np.float64).reshape((6, 32, 32))
    linear_corr = linear_corr[:,10:22,10:22]
    t5 = time.time()
    if DO_MASK:
        chopped_FGS1 = mask_hot_dead(chopped_FGS1, dead_fgs1, dark)
        FGS1_clean[0] = chopped_FGS1
    else:
        FGS1_clean[0] = chopped_FGS1
    t6 = time.time()
    if DO_THE_NL_CORR: 
        linear_corr_signal = apply_linear_corr(linear_corr,FGS1_clean[0])
        FGS1_clean[0,:, :, :] = linear_corr_signal
    del linear_corr
    t7 = time.time()
    if DO_DARK: 
        cleaned_signal = clean_dark(FGS1_clean[0], dead_fgs1, dark,dt_fgs1)
        FGS1_clean[0] = cleaned_signal
    else: 
        pass
    del dark 
    t8 = time.time()
    AIRS_cds = get_cds(AIRS_CH0_clean)
    FGS1_cds = get_cds(FGS1_clean)
    t9 = time.time()
    ## (Optional) calc diff of data between two timestamps
    if TIME_BINNING:
        AIRS_cds_binned = bin_obs(AIRS_cds,binning=30)
        FGS1_cds_binned = bin_obs(FGS1_cds,binning=30*12)
    else:
        AIRS_cds = AIRS_cds.transpose(0,1,3,2) ## this is important to make it consistent for flat fielding, but you can always change it
        AIRS_cds_binned = AIRS_cds
        FGS1_cds = FGS1_cds.transpose(0,1,3,2)
        FGS1_cds_binned = FGS1_cds
    del AIRS_cds, FGS1_cds
    flat_airs = pd.read_parquet(os.path.join(path_folder,f'train/{planet_id}/AIRS-CH0_calibration/flat.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
    flat_fgs = pd.read_parquet(os.path.join(path_folder,f'train/{planet_id}/FGS1_calibration/flat.parquet')).values.astype(np.float64).reshape((32, 32))
    flat_fgs = flat_fgs[10:22,10:22]
    if DO_FLAT:
        corrected_AIRS_cds_binned = correct_flat_field(flat_airs,dead_airs, AIRS_cds_binned[0])
        AIRS_cds_binned[0] = corrected_AIRS_cds_binned
        corrected_FGS1_cds_binned = correct_flat_field(flat_fgs,dead_fgs1, FGS1_cds_binned[0])
        FGS1_cds_binned[0] = corrected_FGS1_cds_binned
    else:
        pass
    AIRS_cds_original = np.expand_dims(get_cds_origial(airs_original), axis=0).transpose(0,1,3,2)
    FGS1_cds_original = np.expand_dims(get_cds_origial(fgs_original), axis=0).transpose(0,1,3,2)
    t10 = time.time()
    deltaT = t10-t0
    if 0:
        print('t1',(t1-t0)/deltaT)
        print('t2',(t2-t1)/deltaT)
        print('t3',(t3-t2)/deltaT)
        print('t4',(t4-t3)/deltaT)
        print('t5',(t5-t4)/deltaT)
        print('t6',(t6-t5)/deltaT)
        print('t7',(t7-t6)/deltaT)
        print('t8',(t8-t7)/deltaT)
        print('t9',(t9-t8)/deltaT)
        print('t10',(t10-t9)/deltaT)
        print(deltaT)
    return AIRS_cds_binned, FGS1_cds_binned,AIRS_cds_original, FGS1_cds_original

def score(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        row_id_column_name: str,
        naive_mean: float,
        naive_sigma: float,
        sigma_true: float
    ) -> float:
    '''
    This is a Gaussian Log Likelihood based metric. For a submission, which contains the predicted mean (x_hat) and variance (x_hat_std),
    we calculate the Gaussian Log-likelihood (GLL) value to the provided ground truth (x). We treat each pair of x_hat,
    x_hat_std as a 1D gaussian, meaning there will be 283 1D gaussian distributions, hence 283 values for each test spectrum,
    the GLL value for one spectrum is the sum of all of them.

    Inputs:
        - solution: Ground Truth spectra (from test set)
            - shape: (nsamples, n_wavelengths)
        - submission: Predicted spectra and errors (from participants)
            - shape: (nsamples, n_wavelengths*2)
        naive_mean: (float) mean from the train set.
        naive_sigma: (float) standard deviation from the train set.
        sigma_true: (float) essentially sets the scale of the outputs.
    '''

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if submission.min().min() < 0:
        raise ParticipantVisibleError('Negative values in the submission')
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')

    n_wavelengths = len(solution.columns)
    if len(submission.columns) != n_wavelengths*2:
        raise ParticipantVisibleError('Wrong number of columns in the submission')

    y_pred = submission.iloc[:, :n_wavelengths].values
    # Set a non-zero minimum sigma pred to prevent division by zero errors.
    sigma_pred = np.clip(submission.iloc[:, n_wavelengths:].values, a_min=10**-15, a_max=None)
    y_true = solution.values

    GLL_pred = np.sum(scipy.stats.norm.logpdf(y_true, loc=y_pred, scale=sigma_pred))
    GLL_true = np.sum(scipy.stats.norm.logpdf(y_true, loc=y_true, scale=sigma_true * np.ones_like(y_true)))
    GLL_mean = np.sum(scipy.stats.norm.logpdf(y_true, loc=naive_mean * np.ones_like(y_true), scale=naive_sigma * np.ones_like(y_true)))

    submit_score = (GLL_pred - GLL_mean)/(GLL_true - GLL_mean)
    return float(np.clip(submit_score, 0.0, 1.0))