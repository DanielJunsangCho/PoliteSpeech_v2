from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import subprocess, platform

def get_reference(audio_file):
    fpath = Path(audio_file)
    wav = preprocess_wav(fpath)

    encoder = VoiceEncoder()
    reference_embed = encoder.embed_utterance(wav)
    return reference_embed

def remove_questions(audio_file, reference, reference2, reference3, removed_files):
    fpath = Path(audio_file)
    wav = preprocess_wav(fpath)

    encoder = VoiceEncoder()
    embed = encoder.embed_utterance(wav)
    score = np.dot(reference, embed) / (np.linalg.norm(reference) * np.linalg.norm(embed))
    score2 = np.dot(reference2, embed) / (np.linalg.norm(reference2) * np.linalg.norm(embed))
    score3 = np.dot(reference3, embed) / (np.linalg.norm(reference3) * np.linalg.norm(embed))
    print(f"{audio_file}: {score}")

    if score > 0.90 or score2 > 0.90 or score3 > 0.90:       #check if voice is the same as female / male interviewer
        removed_files.append(audio_file)
        command = f'rm {audio_file}'
        subprocess.call(command, shell=platform.system() != 'Windows')

    return removed_files

    
