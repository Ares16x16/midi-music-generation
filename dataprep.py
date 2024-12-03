from miditok import REMI, TokenizerConfig
from miditoolkit import MidiFile
from pathlib import Path
from glob import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import *


class DataPreparer:
    def __init__(self, midi_dir, encoding_dir, tokenizer_config):
        self.midi_dir = midi_dir
        self.encoding_dir = encoding_dir
        self.tokenizer = REMI(tokenizer_config)
        os.makedirs(self.encoding_dir, exist_ok=True)

    def convert_and_save(self):
        files = glob(os.path.join(self.midi_dir, "*/*/*/*.mid"))
        print("All Midi files: ", len(files))

        df = []
        error_files = []
        for f in tqdm(files, total=len(files)):
            try:
                midi = MidiFile(f)
                # Handle key signature decoding errors
                try:
                    tokens = list(self.tokenizer(midi))
                except Exception as e:
                    print(f"Error decoding key signature in {f}: {e}")
                    error_files.append(f)
                    continue
                tokens = np.array(tokens)
                savefilename = os.path.basename(f)[:-4] + ".npy"
                savefilepath = os.path.join(self.encoding_dir, savefilename)
                with open(savefilepath, "wb") as f_:
                    np.save(f_, tokens)
                df.append([os.path.basename(savefilepath), tokens.shape[0]])
            except Exception as e:
                print(f"Error processing {f}: {e}")
                error_files.append(f)
                continue

        df = pd.DataFrame(df, columns=["fname", "tokenLength"])
        df.to_csv(os.path.join(self.encoding_dir, "df.csv"), index=False)
        self.tokenizer.save(self.encoding_dir)
        print("Vocab size: ", self.tokenizer.len)

        # Save the names of the error files to a text file
        with open(
            os.path.join(self.encoding_dir, "error_files.txt"), "w"
        ) as error_file:
            for ef in error_files:
                error_file.write(f"{ef}\n")


def main():
    tokenizer_config = TokenizerConfig(
        use_programs=True, one_token_stream_for_programs=True, use_time_signatures=True
    )
    preparer = DataPreparer(MIDI_DIR, ENCODING_DIR, tokenizer_config)
    preparer.convert_and_save()


if __name__ == "__main__":
    main()
