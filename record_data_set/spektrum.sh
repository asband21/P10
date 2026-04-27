
find . -name "chunk_*.wav" -exec sox {} -n spectrogram -o {}.png \;

