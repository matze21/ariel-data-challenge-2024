from conversion_helpers import *

import pandas as pd
import os
import numpy as np
import multiprocessing

def encodeDataTop(row):
    planet_id,train_adc_info,axis_info,DO_MASK,DO_THE_NL_CORR,DO_DARK,DO_FLAT,TIME_BINNING = row
    encodeData(planet_id,train_adc_info,axis_info,DO_MASK,DO_THE_NL_CORR,DO_DARK,DO_FLAT,TIME_BINNING)

def encodeData(planet_id,train_adc_info,axis_info,DO_MASK,DO_THE_NL_CORR,DO_DARK,DO_FLAT,TIME_BINNING, zScoreNorm=True):
    path_folder = ""  

    if os.path.exists('train/'+str(planet_id)+'/combined.npz'):
        print(planet_id,' already exists')
    else:
        AIRS_cds_binned, FGS1_cds_binned,AIRS_cds_original, FGS1_cds_original = calibrateData(planet_id,train_adc_info,axis_info,DO_MASK,DO_THE_NL_CORR,DO_DARK,DO_FLAT,TIME_BINNING)

        if zScoreNorm:
            mean = np.mean(AIRS_cds_binned, axis=(1,2,3), keepdims=True)
            std = np.std(AIRS_cds_binned, axis=(1,2,3), keepdims=True)
            zScoreAIRSPlanet = (AIRS_cds_binned - mean) / std

            mean = np.mean(FGS1_cds_binned, axis=(1,2,3), keepdims=True)
            std = np.std(FGS1_cds_binned, axis=(1,2,3), keepdims=True)
            zScoreFGS1Planet = (FGS1_cds_binned - mean) / std


            mean = np.mean(AIRS_cds_binned, axis=(1,2), keepdims=True)
            std = np.std(AIRS_cds_binned, axis=(1,2), keepdims=True)
            zScoreAIRSWaveL = (AIRS_cds_binned - mean) / std
            mean = np.mean(FGS1_cds_binned, axis=(1,2), keepdims=True)
            std = np.std(FGS1_cds_binned, axis=(1,2), keepdims=True)
            zScoreFGS1WaveL = (FGS1_cds_binned - mean) / std
        else:
            min = np.min(AIRS_cds_binned, axis=(1,2,3), keepdims=True)
            max = np.max(AIRS_cds_binned, axis=(1,2,3), keepdims=True)
            zScoreAIRSPlanet = (AIRS_cds_binned - min) / (max-min)

            min = np.min(FGS1_cds_binned, axis=(1,2,3), keepdims=True)
            max = np.max(FGS1_cds_binned, axis=(1,2,3), keepdims=True)
            zScoreFGS1Planet = (FGS1_cds_binned - min) / (max-min)


            min = np.min(AIRS_cds_binned, axis=(1,2), keepdims=True)
            max = np.max(AIRS_cds_binned, axis=(1,2), keepdims=True)
            zScoreAIRSWaveL = (AIRS_cds_binned - min) / (max-min)

        # compress data, cleanedl
        AIRS_cds_cleaned_compressed = AIRS_cds_binned.sum(axis=3)  # 1x5625x282
        FGS1_cds_cleaned_compressed = FGS1_cds_binned.sum(axis=(2,3)) # 1x67500
        FGS1_cds_cleaned_compressed = np.reshape(FGS1_cds_cleaned_compressed, (1,5625,-1)) #1x5625x12

        del AIRS_cds_binned, FGS1_cds_binned

        # compress original data
        AIRS_cds_original_compressed = AIRS_cds_original.sum(axis=3)  # 1x5625x282
        FGS1_cds_original_compressed = FGS1_cds_original.sum(axis=(2,3)) # 1x67500
        FGS1_cds_original_compressed = np.reshape(FGS1_cds_original_compressed, (1,5625,-1)) #1x5625x12

        del AIRS_cds_original, FGS1_cds_original

        # compress normalized data by planet
        AIRS_cds_PlanetNorm_compressed = zScoreAIRSPlanet.sum(axis=3)  # 1x5625x282
        FGS1_cds_PlanetNorm_compressed = zScoreFGS1Planet.sum(axis=(2,3)) # 1x67500
        FGS1_cds_PlanetNorm_compressed = np.reshape(FGS1_cds_PlanetNorm_compressed, (1,5625,-1)) #1x5625x12

        del zScoreAIRSPlanet, zScoreFGS1Planet

        # compress normlized data by wavelength
        AIRS_cds_WaveLNorm_compressed = zScoreAIRSWaveL.sum(axis=3)  # 1x5625x282
        FGS1_cds_WaveLNorm_compressed = zScoreFGS1WaveL.sum(axis=(2,3)) # 1x67500
        FGS1_cds_WaveLNorm_compressed = np.reshape(FGS1_cds_WaveLNorm_compressed, (1,5625,-1)) #1x5625x12

        del zScoreAIRSWaveL, zScoreFGS1WaveL

        compressed_clean = np.concatenate([AIRS_cds_cleaned_compressed,np.sum(FGS1_cds_cleaned_compressed, axis=2, keepdims=True),np.mean(FGS1_cds_cleaned_compressed, axis=2, keepdims=True),np.std(FGS1_cds_cleaned_compressed, axis=2, keepdims=True)], axis=2)
        del AIRS_cds_cleaned_compressed, FGS1_cds_cleaned_compressed
        compressed_origi = np.concatenate([AIRS_cds_original_compressed,np.sum(FGS1_cds_original_compressed, axis=2, keepdims=True),np.mean(FGS1_cds_original_compressed, axis=2, keepdims=True),np.std(FGS1_cds_original_compressed, axis=2, keepdims=True)], axis=2)
        del AIRS_cds_original_compressed, FGS1_cds_original_compressed
        compressed_plNor = np.concatenate([AIRS_cds_PlanetNorm_compressed,np.sum(FGS1_cds_PlanetNorm_compressed, axis=2, keepdims=True),np.mean(FGS1_cds_PlanetNorm_compressed, axis=2, keepdims=True),np.std(FGS1_cds_PlanetNorm_compressed, axis=2, keepdims=True)], axis=2)
        del AIRS_cds_PlanetNorm_compressed, FGS1_cds_PlanetNorm_compressed
        compressed_waNor = np.concatenate([AIRS_cds_WaveLNorm_compressed,np.sum(FGS1_cds_WaveLNorm_compressed, axis=2, keepdims=True),np.mean(FGS1_cds_WaveLNorm_compressed, axis=2, keepdims=True),np.std(FGS1_cds_WaveLNorm_compressed, axis=2, keepdims=True)], axis=2)
        del AIRS_cds_WaveLNorm_compressed, FGS1_cds_WaveLNorm_compressed

        combined_array = np.stack([compressed_clean,compressed_origi,compressed_plNor,compressed_waNor], axis=-1)

        np.savez('train/'+str(planet_id)+'/combined.npz', a=combined_array)
        print('finished ', planet_id)



if __name__ == '__main__':
    # ATTENTION: this is using 8 cores by default, if you run locally or not a lot of cores are available, change this
    # when setting cores to 1 or smaller the fallback version is used with a single thread
    cores = 7
    path_folder = ""  
    train_adc_info = pd.read_csv(os.path.join(path_folder, 'train_adc_info.csv'))
    train_adc_info = train_adc_info.set_index('planet_id')
    axis_info = pd.read_parquet(os.path.join(path_folder,'axis_info.parquet'))

    DO_MASK = True  # filter out non responsive pixels
    DO_THE_NL_CORR = True # most time consuming step, you can choose to ignore it for rapid prototyping, nonlinear correction due to artefacts when reading pixels
    DO_DARK = True  # dark current is accumulating over time in the pixels, need to compensate that (seems like integration artefact)
    DO_FLAT = True  # pixel to pixel variation correction (e.g. how pixels respond differently when illuminated uniformly)
    TIME_BINNING = False  #do a time binning on choosen frequency


    pool = multiprocessing.Pool(cores)

    files = glob.glob(os.path.join('train/', '*/*'))
    stars = []
    for file in files:
        file_name = file.split('\\')[1]
        stars.append(file_name)
    stars = np.unique(stars)

    input = []
    for i, star in enumerate(stars):
        input.append([int(star),train_adc_info,axis_info,DO_MASK,DO_THE_NL_CORR,DO_DARK,DO_FLAT,TIME_BINNING])
    print('processing inputs: ', len(input))
    results = list(pool.map(encodeDataTop, input))