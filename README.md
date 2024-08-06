# Real-time-voice-cloner
___

`Real-time-voice-cloner` is a open source project contributed by Corentin Jemine. Find the link to the repo <a href = "https://github.com/CorentinJ/Real-Time-Voice-Cloning">here<a>.
#### **Goal of this project-**

We attempted to implement the project from scratch with the corentin J's repo as reference.
Working on this has been a great learning experience in the fields of Audio signal processing and deep learning technologies.

We use Jupyter notebooks for preprocessing, model training and inference as it helps in better understanding the flow of things.
___
#### **About the system -** 

The system - voice cloner can be better dubbed as a multi-speaker Text To Speech (TTS) system. Its main capablity is to generate speech from text in the voice of the provided target speaker with only 5 seconds of audio sample.
Traditional TTS systems have beeen built by training their models on huge volumes of transcribed speech data, which  usually is a very costly affair.

The project is a implementation research paper - <a href = "https://arxiv.org/pdf/1806.04558.pdf"> Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis (SV2TTS)<a>.
The paper proproses an encoder system based on d-vectors used for speaker recoginition activities to generate embeddings on the audio samples.
These embeddings capture the voice characterteristics of the speaker and helps tell apart different speakers from one another.


The system consists of three main components 

* `encoder` - Sub-system to generate the  embeddings of audio samples.
* `synthesizer` - This is the heart of the system that generates audio waveform of target speaker corresponding to the provided text.
* `vocoder` - This regenerates natural human like speech audio. 







