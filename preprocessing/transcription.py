import os
from dotenv import load_dotenv

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

load_dotenv()


def createDataset(dataset, audio_files, api_key="da81c42d95b0bb82c02b43b36002b215ccb0039c"):
    try:
        deepgram = DeepgramClient(api_key)
        TIMEOUT = 300   #5 minute timeout

        for audio_file in audio_files:
            print(f"loading {audio_file} to dataset...")
            with open(audio_file, "rb") as file:
                buffer_data = file.read()

            payload: FileSource = {
                "buffer": buffer_data,
            }

            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
                filler_words=True,      #to enable filler words
            )

            response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options, timeout=TIMEOUT)
            transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
            dataset[audio_file] = transcript
            
        return dataset
            

    except Exception as e:
        print(f"Exception: {e}")


def checkFalsePositives(dataset, audio_labels):
    for key in dataset.keys():
        transcription = dataset[key]
        if transcription.endswith('.') or transcription.endswith('?') or transcription.endswith('!'):
            continue
        else:
            audio_labels[key] = 0
    return dataset, audio_labels
