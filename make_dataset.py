import preprocessing.audio as Audio
import preprocessing.transcription as Transcript
import preprocessing.remove_interviewer as Trim
import sys, os
import pandas as pd
import numpy as np
import csv
    


def main():
	directories = [
	    "samples/initial_samples",
	    "samples/stereo_samples",
	    "samples/speaker_endpoints",
	    "samples/interviewer_samples"
	]
	
	for directory in directories:
	    os.makedirs(directory, exist_ok=True)
	
	path = "samples/initial_samples"
	
	# create initial dataset by splitting by silence and getting last seven seconds of each chunk of audio
	for audio_file in os.listdir(path):
	if audio_file != ".DS_Store":
	    print(f"Converting {audio_file}...")
	
	    audio_file_path = os.path.join(path, audio_file)
	    output_file_path = os.path.join("samples/stereo_samples", audio_file)
	    Audio.convert_wav(audio_file_path, output_file_path)
	
	print("--------------------------------------------------------- \n STARTING DIARIZATION \n --------------------------------------------------------- ")
	for audio_file in os.listdir("samples/stereo_samples"):
	print(f"Splitting: {audio_file}...")
	audio_file = os.path.join("samples/stereo_samples", audio_file)
	
	speaker_changes = Audio.diarization(audio_file)
	Audio.splitBySpeakers(audio_file, speaker_changes)
	
	audio_files = []
	for audio_file in os.listdir("samples/speaker_endpoints"):
	audio_file = os.path.join("samples/speaker_endpoints", audio_file)
	chunk_labels = Audio.splitBySilence(audio_file)         # => split speaker audio by thinking pauses (look at function for chunk parameter specifications)
	audio_files.append(chunk_labels)
	
	audio_labels = {}
	for array in audio_files:
	for element in array:
	    audio_labels[element[0]] = element[1]
	
	prelim_files = list(audio_labels.keys())
	
	print("--------------------------------------------------------- \n REMOVING INTERVIEWERS\n --------------------------------------------------------- ")
	# #make references to compare voice identities to remove interviewers from dataset
	reference_path = "samples/interviewer_samples/female_combined.wav"
	reference_path2 = "samples/interviewer_samples/male_combined.wav"
	reference_path3 = "samples/interviewer_samples/male2_combined.wav"
	reference = Trim.get_reference(reference_path)
	reference2 = Trim.get_reference(reference_path2)
	reference3 = Trim.get_reference(reference_path3)
	
	# #compare each audio with interviewer voice identities, remove the audio containing interviewer audio
	removed_files = []
	for audio_file in prelim_files:
	removed_files = Trim.remove_questions(audio_file, reference, reference2, reference3, removed_files)
	
	#confirm final audio datset
	final_files = [file for file in prelim_files if file not in removed_files]
	
	
	# export audio file, transcription, label to csv dataset
	print("--------------------------------------------------------- \n TRANSCRIBING AUDIO\n --------------------------------------------------------- ")
	dataset = Transcript.createDataset({}, final_files) 
	
	dataset = {}
	audio_labels = {}
	
	
	
	# Open and read the CSV file
	with open('datasets/final_dataset.csv', 'r') as csvfile:
	reader = csv.DictReader(csvfile)
	
	# Iterate through each row in the CSV
	for row in reader:
	    audio_file = row['audio file']
	    transcription = row['transcription']
	    label = row['label']
	
	    # Populate the dictionaries
	    dataset[audio_file] = transcription
	    audio_labels[audio_file] = label
	
	
	dataset, audio_labels = Transcript.checkFalsePositives(dataset, audio_labels)
	
	
	Audio.exportDataset(dataset, audio_labels)


    


if __name__ == '__main__':
	main()
