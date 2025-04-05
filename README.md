# Speaker Endpoint Detection Model

## Overview
This repository encompasses a personal curiosity project that I developed while talking with my friends: how does Siri know when I'm done talking?
The following outlines the training process of a deep learning, binary classification model capable of detecting when an individual has completely finished their train of thought,     
     regardless of pauses stemming from a broken train of thought, end of sentence, a cough, hack, hem, or whoop. 
First, audio files from the dataset are split into 5 second chunks and stored, along with their transcriptions. Then the model is trained with these 5 second audio chunks and 
related transcription. The model's hyperparameters are automatically tuned using Optuna's experimentation tool.

## Prerequisites
* pyannote
* librosa
* wave
* soundfile
* resemblyzer
* pathlib
* deepgram
* optuna

## Steps to Train Model

**Data Preprocessing**
1. Load all audio files into samples/initial_samples
2. python make_dataset.py

At this point, there should be a csv called datasets/final_dataset.csv, which holds all necessary information for training. All processed audio files should be in samples/processed_samples.

**Model Training**
1. In Terminal, run 'mlflow server'
2. python experiment_model.py [path to processed samples]

Note: Before you can run the model experiment script, you will want to ensure that your local is running MLFlow, as mentioned in Step 1. You may want to change this line in 
experiment_model.py: mlflow.set_tracking_uri("http://127.0.0.1:5000/")

**Saving Model as .pth File**
1. In lines 19-24 of inference.py, there are lines commented out that pull a model from the experiment tracker and saves the PyTorch model as a .pth file. These .pth files are how predictions
are typically done, allowing for quick predictions.
2. In the lines mentioned above, change the tracking uri and the logged model id based on the model you want to save.
3. Once 'model.pth' has been created, you can comment out lines 19-24 and run inference.py to ensure that the model can run with the processed inputs.

Now you can use this .pth file/model to shadow interviews.


