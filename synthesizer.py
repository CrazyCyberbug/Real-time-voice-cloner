from synthesizer.preprocess import * 
from synthesizer.train import train
from pathlib import Path
from paths import *
import encoder.inference2 as encoder
from synthesizer.generate_embeds import embed_utterances

model_path = models_dir.joinpath("synthesizer")
model_path.mkdir(exist_ok= True)
syn_outdir = Path("./syn_out2")

if __name__ == "__main__":
    
    # preprocess
    dataset_root = Path("C:/Users/Swaroop/clean-100-with-alignments/train-clean-100")
    out_dir = Path("synthesizer/outdir3")
    out_dir.mkdir(exist_ok= True)
    preprocess_dataset(dataset_root, out_dir)
    
    #create embeddings
   
    embed_utterances(Path("synthesizer"), Path("trained-models/encoder.pt"))
      

    #train
    
    train("1", syn_dir = Path("./synthesizer"), models_dir = Path("./trained-models/synthesizer"), save_every=10, force_restart= False)