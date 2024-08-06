from vocoder.vocoder_dataset import VocoderDataset
from pathlib import Path
from vocoder.train import train
from paths import *


if __name__ == '__main__':
    
   
    model_dir = models_dir.joinpath("vocoder")
    model_dir.mkdir(exist_ok = True)

    metadata_fpath = Path("C:/Users/Swaroop/real-time-voice-cloner/synthesizer/outdir/train.txt")
    mel_dir = Path("C:/Users/Swaroop/real-time-voice-cloner/synthesizer/outdir/mels")
    wav_dir = Path("C:/Users/Swaroop/real-time-voice-cloner/synthesizer/outdir/audio")

    dataset = VocoderDataset(metadata_fpath, mel_dir, wav_dir)

    print("starting trainaing...")
    model_dir = models_dir.joinpath("vocoder")
    train(run_id = "1", syn_dir= Path("synthesizer/"), voc_dir=Path("vocoder"), models_dir=model_dir, ground_truth=True, force_restart=False)