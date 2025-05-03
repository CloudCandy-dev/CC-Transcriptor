from faster_whisper import WhisperModel
import os

# @Requires
# To run in cuda (fast) mode, the cuda library, cudnn library and pytorch are required.

# Set input and output folder paths.
current_path = os.getcwd()
input_folder = os.path.join(current_path, "input_mp3")
output_folder = os.path.join(current_path, "output_txt")
print(f"Input folder: {input_folder}")
print(f"Output folder: {output_folder}")

# Create output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# set model type
model_size = "large-v3-turbo"

# run mode setting
# if you have the environment is cuda-enabled
model = WhisperModel(model_size, device="cuda", compute_type="float16")
# if you don't
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Process all MP3 files in the input_mp3 folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".mp3"):
        input_file_path = os.path.join(input_folder, file_name)
        print(f"Processing: {input_file_path}")
        
        # Transcribing audio files
        segments, info = model.transcribe(audio=input_file_path, beam_size=5, language="ja")
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        # Set output file path.
        output_file_name = os.path.splitext(file_name)[0] + "_transcript" + ".txt"
        output_file_path = os.path.join(output_folder, output_file_name)
        
        # Save the results in a text file.
        with open(output_file_path, "a", encoding="utf-8") as output_file:
            for segment in segments:
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text),file=output_file)
        
        print(f"Transcription saved to: {output_file_path}")
