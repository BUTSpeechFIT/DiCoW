import wave
import struct
import random

def generate_dummy_wav(filename="dummy.wav", duration_seconds=10, sample_rate=16000):
    print(f"Generating {filename} with noise...")
    n_frames = duration_seconds * sample_rate
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for _ in range(n_frames):
            # Add some white noise
            value = random.randint(-500, 500)
            data = struct.pack('<h', value)
            wav_file.writeframesraw(data)
    
    print(f"Done. Created {filename}")

if __name__ == "__main__":
    generate_dummy_wav()
