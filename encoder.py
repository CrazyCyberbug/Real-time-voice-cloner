# from encoder import params_data, params_model

from encoder.audio import *
from encoder.preprocess import _preprocess_speaker
from encoder.model2 import SpeakerEncoder
from encoder.train import train

# from encoder.paths import *

import numpy as np
import librosa
from pathlib import Path
from typing import Optional, Union
import torch
from paths import *


import warnings
warnings.filterwarnings("ignore")



#paths 
dataset_root = Path("C:/Users/Swaroop/Downloads/train-clean-100/LibriSpeech/train-clean-100")
outdir = Path("./outdir4")
outdir.mkdir(exist_ok= True)
models_dir = Path("./trained-models2")
models_dir.mkdir(exist_ok = True)  




# helper function to apply the preprocessing to all the spekaer directories in the dataset.

def apply_preprocessing_to_all_speakers(data_set_root: Path, output_dir: Path):
    speakers = list(data_set_root.glob("*"))
    for speaker in speakers:
        _preprocess_speaker(speaker, datasets_root= dataset_root, out_dir=outdir)     
        
        
        
# code execution starts here!
if __name__ == "__main__":
    
    dataset_root = Path("C:/Users/Swaroop/Downloads/train-clean-100/LibriSpeech/train-clean-100")
    outdir = Path("./outdir4")
    outdir.mkdir(exist_ok= True)
    models_dir = Path("./trained-models2")
    models_dir.mkdir(exist_ok = True)
    
    if dataset_root.exists() and dataset_root.is_dir():
        print("dataset ", dataset_root.name, "found")
    else:
        raise Exception("dataset not found! please check your dataset root path!")    
    
    
    #step 1  preprocess the audio file!
    # if the files are already preprocessed, just comment out the following code step!
    
    print("preprocessing your data!")
    
    apply_preprocessing_to_all_speakers(dataset_root, outdir)
    
    print("data preprocessed!")
    
    
    # step 2: model trainaing!
    # you can either start the trainaing from start or from restart training a pretrained model (use the forece_reastart param in the function definition)
    print("start training...")
    model = SpeakerEncoder(torch.device('cpu'), torch.device('cpu'))
    
    encoder_model_dir = models_dir.joinpath("encoder")
    encoder_model_dir.mkdir(exist_ok = True)
    new_models_dir = Path('./new_model_dir/')
    new_models_dir.mkdir(exist_ok=True)
    train("1", outdir, encoder_model_dir, 10, 10, 10) # the string that says 1 is to mark the  version of the model such that you can traina ad haVE  multiple versions of the modl
    
   
    
        






