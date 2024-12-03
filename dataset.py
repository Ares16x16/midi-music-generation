import os
import numpy as np
import torch
from torch.utils.data import Dataset
from config import ENCODING_DIR, SEQ_LEN
import logging


class MidiDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.indices = self._create_indices()

    def _create_indices(self):
        indices = {}
        global_index = 0

        if self.df.empty:
            logging.error("The DataFrame is empty.")
            raise ValueError("The DataFrame is empty.")

        for file_index, row in self.df.iterrows():
            fname = os.path.join(ENCODING_DIR, row["fname"])
            if not os.path.isfile(fname):
                logging.error("File does not exist: {}".format(fname))
                continue

            with open(fname, "rb") as f:
                nparray = np.load(f)

            for array_index in range(nparray.shape[0] - SEQ_LEN):
                indices[global_index] = {"file": fname, "array_index": array_index}
                global_index += 1

        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        fname = self.indices[idx]["file"]
        array_index = self.indices[idx]["array_index"]

        with open(fname, "rb") as f:
            nparray = np.load(f)

        x = torch.tensor(nparray[array_index : array_index + SEQ_LEN], dtype=torch.long)
        y = torch.tensor(
            nparray[array_index + 1 : array_index + SEQ_LEN + 1], dtype=torch.long
        )

        return {"x": x, "y": y}
