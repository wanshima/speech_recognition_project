from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

# Get the text input from the user
text = input("Enter the text you want to synthesize: ")

# Define the model ID
model_id = 'damo/speech_sambert-hifigan_tts_zh-cn_16k'

# Create a pipeline for text-to-speech
sambert_hifigan_tts = pipeline(task=Tasks.text_to_speech, model=model_id)

# Generate speech from the text
output = sambert_hifigan_tts(input=text, voice='zhitian_emo')

# Extract the generated wav file from the output
wav = output[OutputKeys.OUTPUT_WAV]

# Save the generated speech to a file
with open('output.wav', 'wb') as f:
    f.write(wav)

print("Speech synthesis complete. The output is saved as 'output.wav'.")