DEVICE = "cuda"  #'cuda'
EPOCHS = 10
BATCH_SIZE = 20
SEQ_LEN = 64
VOCAB_SIZE = 484
LOG_DIR = "output/logs"
TRAIN_TEST_RATIO = 0.9

EMBED_DIM = 768
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 1
LR = 9e-6

SUFFIX = "adl_remi_plus"
MODEL_DIR = "output/model_{}".format(SUFFIX)
ENCODING_DIR = "encoded_adl_midi"
# ENCODING_DIR = "encoded_adl_midi_mini"
MIDI_DIR = "adl-piano-midi"
