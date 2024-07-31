from pyannote.audio import Pipeline
from audio_slicer.slicer2 import Slicer
from pydub import AudioSegment
import torch
import numpy as np
from scipy.io import wavfile
import librosa, wave
import soundfile as sf
import os, csv, shutil, subprocess
import time

def diarization(wav_file):
    start = time.perf_counter()

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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")

    # send pipeline to GPU (when available)
    pipeline.to(torch.device("cpu"))

    # apply pretrained pipeline
    diarization = pipeline(wav_file)

    # split by when speakers change in audio
    last_speaker = ""
    last_end = ""
    speaker_changes = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if last_speaker != "":
            if last_speaker != speaker:
                last_speaker = speaker
                speaker_changes.append(last_end)
        else:
            last_speaker = speaker
        last_end = turn.end
    print(f"{wav_file} took {time.perf_counter() - start} seconds to split by speaker")
    return speaker_changes

def splitBySpeakers(wav_file, speaker_changes):
    audio = AudioSegment.from_wav(wav_file)
    file_id = wav_file.split('/')[2].split('.')[0]

    start = 0
    for i, t in enumerate(speaker_changes):
        end = t * 1000 #pydub works in millisec
        audio_chunk = audio[start:end]
        silence = AudioSegment.silent(duration=1000)
        audio_chunk = audio_chunk + silence #add silence to the end of speaker chunk to ensure the last "endpoint" gets labeled as one
        audio_chunk.export(f"samples/speaker_endpoints/{file_id}_{i}.wav", format="wav")

        start = end

    destination = os.path.join("samples/used", os.path.basename(wav_file))
    shutil.move(wav_file, destination)

def splitBySilence(wav_file, cutoff=5, threshold=-40, min_length=7000, min_interval=800, hop_size=10, max_sil_kept=800):
    # Load the audio file
    print(f"Splitting {wav_file} by silence...")
    audio, sr = librosa.load(wav_file, sr=None, mono=False)
    file_id = wav_file.split('/')[2].split('.')[0]

    slicer = Slicer(
        sr=sr,
        threshold=threshold,
        min_length=min_length,
        min_interval=min_interval,
        hop_size=hop_size,
        max_sil_kept=max_sil_kept,
    )

    # Slice the audio
    chunks = slicer.slice(audio)

    # Calculate total duration and the 1.5 second threshold
    total_duration = len(audio[0]) / sr
    threshold_time = total_duration - 1

    cutoff_length = cutoff * sr
    chunk_labels = []
    for i, (chunk, start_time, end_time) in enumerate(chunks): 
        chunk = chunk.T
        chunk_length = chunk.shape[0]

        if chunk_length > cutoff_length:
            chunk = chunk[-cutoff_length:]

            output_file = os.path.join("samples/processed_samples", f"{file_id}_{i}.wav")
            sf.write(output_file, chunk, sr)

            if end_time / sr > threshold_time:
                print(end_time / sr, threshold_time)
                chunk_labels.append((output_file, 1))
            else:
                chunk_labels.append((output_file, 0))
    return chunk_labels


def extract_audio_features(wav_file):
    audio, sr = librosa.load(wav_file, sr=44100, mono=False)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_1d = mfccs.flatten()
    return mfccs_1d


def convert_wav(input_file, output_file):
    # Open the input WAV file
    with wave.open(input_file, 'rb') as wav_in:
        # Read the WAV file data
        params = wav_in.getparams()
        numchannels, sampwidth, framerate, nframes, comptype, compname = params

        if numchannels == 1:
            print(f"Converting {input_file} file mono to stereo...")

            with wave.open(output_file, 'wb') as wav_out:
                # Set the parameters for the output file, changing nchannels to 2
                wav_out.setparams((2, sampwidth, framerate, nframes, comptype, compname))

                # Read and write the frames
                for _ in range(nframes):
                    frame = wav_in.readframes(1)
                    wav_out.writeframes(frame * 2)  # Duplicate the frame for stereo

        else:
            shutil.copy2(input_file, output_file)


def mp3_to_wav(audio_file):
    file_id = audio_file.split('.')[0].split('/')[2]
    ffmpeg_command = [
        "ffmpeg",
        "-i", audio_file,
        "-acodec", "pcm_s16le",
        "-ac", "2",
        f"samples/initial_samples/{file_id}.wav"
    ]
    
    subprocess.run(ffmpeg_command)
 
def exportDataset(dataset, audio_labels):
    with open("datasets/final_dataset.csv", 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["audio file", "transcription", "label"])

        files = list(dataset.keys())
        transcriptions = list(dataset.values())
        for file, transcription in zip(files, transcriptions):
            csvwriter.writerow([file, transcription, audio_labels[file]])
            