import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import glob
import os

import itertools
from astropy.stats import sigma_clip

from tqdm import tqdm

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

    cut_inf, cut_sup = 39, 321
    l = cut_sup - cut_inf

    AIRS_CH0_clean = np.zeros((1, 11250, 32, l))
    FGS1_clean = np.zeros((1, 135000, 32, 32))

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

    if DO_MASK:
        chopped_signal = mask_hot_dead(chopped_signal, dead_airs, dark)
        AIRS_CH0_clean[0] = chopped_signal
    else:
        AIRS_CH0_clean[0] = chopped_signal

    if DO_THE_NL_CORR: 
        linear_corr_signal = apply_linear_corr(linear_corr,AIRS_CH0_clean[0])
        AIRS_CH0_clean[0] = linear_corr_signal
    del linear_corr

    if DO_DARK: 
        cleaned_signal = clean_dark(AIRS_CH0_clean[0], dead_airs, dark,dt_airs)
        AIRS_CH0_clean[0] = cleaned_signal
    else: 
        pass
    del dark

    df = pd.read_parquet(os.path.join(path_folder,f'train/{planet_id}/FGS1_signal.parquet'))
    fgs_signal = df.values.astype(np.float64).reshape((df.shape[0], 32, 32))
    FGS1_gain = train_adc_info['FGS1_adc_gain'].loc[planet_id]
    FGS1_offset = train_adc_info['FGS1_adc_offset'].loc[planet_id]
    fgs_signal = ADC_convert(fgs_signal, FGS1_gain, FGS1_offset)
    dt_fgs1 = np.ones(len(fgs_signal))*0.1  ## please refer to data documentation for more information
    dt_fgs1[1::2] += 0.1
    chopped_FGS1 = fgs_signal
    fgs_original = chopped_FGS1
    del fgs_signal, df

    # CLEANING THE DATA: FGS1
    flat = pd.read_parquet(os.path.join(path_folder,f'train/{planet_id}/FGS1_calibration/flat.parquet')).values.astype(np.float64).reshape((32, 32))
    dark = pd.read_parquet(os.path.join(path_folder,f'train/{planet_id}/FGS1_calibration/dark.parquet')).values.astype(np.float64).reshape((32, 32))
    dead_fgs1 = pd.read_parquet(os.path.join(path_folder,f'train/{planet_id}/FGS1_calibration/dead.parquet')).values.astype(np.float64).reshape((32, 32))
    linear_corr = pd.read_parquet(os.path.join(path_folder,f'train/{planet_id}/FGS1_calibration/linear_corr.parquet')).values.astype(np.float64).reshape((6, 32, 32))

    if DO_MASK:
        chopped_FGS1 = mask_hot_dead(chopped_FGS1, dead_fgs1, dark)
        FGS1_clean[0] = chopped_FGS1
    else:
        FGS1_clean[0] = chopped_FGS1

    if DO_THE_NL_CORR: 
        linear_corr_signal = apply_linear_corr(linear_corr,FGS1_clean[0])
        FGS1_clean[0,:, :, :] = linear_corr_signal
    del linear_corr

    if DO_DARK: 
        cleaned_signal = clean_dark(FGS1_clean[0], dead_fgs1, dark,dt_fgs1)
        FGS1_clean[0] = cleaned_signal
    else: 
        pass
    del dark 

    AIRS_cds = get_cds(AIRS_CH0_clean)
    FGS1_cds = get_cds(FGS1_clean)


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
    if DO_FLAT:
        corrected_AIRS_cds_binned = correct_flat_field(flat_airs,dead_airs, AIRS_cds_binned[0])
        AIRS_cds_binned[0] = corrected_AIRS_cds_binned
        corrected_FGS1_cds_binned = correct_flat_field(flat_fgs,dead_fgs1, FGS1_cds_binned[0])
        FGS1_cds_binned[0] = corrected_FGS1_cds_binned
    else:
        pass

    AIRS_cds_original = np.expand_dims(get_cds_origial(airs_original), axis=0).transpose(0,1,3,2)
    FGS1_cds_original = np.expand_dims(get_cds_origial(fgs_original), axis=0).transpose(0,1,3,2)

    return AIRS_cds_binned, FGS1_cds_binned,AIRS_cds_original, FGS1_cds_original

