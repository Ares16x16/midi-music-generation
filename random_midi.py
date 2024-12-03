import numpy as np
from miditoolkit import MidiFile, Note, Instrument
from pathlib import Path

# Define parameters for random note generation
num_notes = 100  # Number of random notes to generate
min_pitch = 21  # Minimum MIDI pitch (A0)
max_pitch = 108  # Maximum MIDI pitch (C8)
min_velocity = 20  # Minimum velocity
max_velocity = 127  # Maximum velocity
min_duration = 120  # Minimum duration in ticks
max_duration = 480  # Maximum duration in ticks
ticks_per_beat = 480  # Ticks per beat (standard value)
total_duration = 480 * 100  # Total duration of the song in ticks

# Create a new MIDI file and an instrument
midi = MidiFile(ticks_per_beat=ticks_per_beat)
instrument = Instrument(program=0, is_drum=False, name="Random Instrument")

# Generate random notes
current_time = 0
for _ in range(num_notes):
    pitch = np.random.randint(min_pitch, max_pitch + 1)
    velocity = np.random.randint(min_velocity, max_velocity + 1)
    duration = np.random.randint(min_duration, max_duration + 1)
    start = current_time
    end = start + duration
    note = Note(velocity=velocity, pitch=pitch, start=start, end=end)
    instrument.notes.append(note)
    current_time += np.random.randint(min_duration, max_duration + 1)
    if current_time >= total_duration:
        break

# Add the instrument to the MIDI file
midi.instruments.append(instrument)

# Save the generated MIDI file
output_path = Path("sample/random_notes/3.mid")
output_path.parent.mkdir(parents=True, exist_ok=True)
midi.dump(output_path)
