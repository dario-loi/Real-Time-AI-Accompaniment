# =========================================================================
# Vocabulary Class for representing chords for the LSTM-based model
#
#   Represents chords as indices for efficient model training
#   0: <PAD>, 1: <UNK>, 2..N: actual chords.
#   Provides a main to build, save, load, and test the vocabulary.
# =========================================================================


import pickle
from collections import Counter
import os

class Vocabulary:
    def __init__(self):
        # we use 2 dicts for fast lookup
        self.chord_to_idx = {}
        self.idx_to_chord = {}

        self.pad_token = "<PAD>"
        self.unknown_token = "<UNK>"

    # builds vocabulary from a list of songs (lists of chords)
    def build_vocab(self, songs, min_freq=1):

        all_chords = [chord for song in songs for chord in song]
        chord_counts = Counter(all_chords)
        
        # sort by frequency then alphabetically for determinism
        sorted_chords = sorted(chord_counts.keys())
        
        # add special tokens first
        self.chord_to_idx = {self.pad_token: 0, self.unknown_token: 1}
        self.idx_to_chord = {0: self.pad_token, 1: self.unknown_token}
        
        idx = 2
        for chord in sorted_chords:
            if chord_counts[chord] >= min_freq:
                self.chord_to_idx[chord] = idx
                self.idx_to_chord[idx] = chord
                idx += 1
                
        return len(self.chord_to_idx)

    # saves vocabulary mappings to a pickle file
    def save_vocab(self, path):

        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'chord_to_idx': self.chord_to_idx,
                'idx_to_chord': self.idx_to_chord
            }, f)

    # loads vocabulary mappings from a pickle file
    def load_vocab(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.chord_to_idx = data['chord_to_idx']
            self.idx_to_chord = data['idx_to_chord']

    # returns the size of the vocabulary
    def __len__(self):
        return len(self.chord_to_idx)

    # converts a list of chords to indices
    def to_indices(self, chords):
        return [self.chord_to_idx.get(c, self.chord_to_idx[self.unknown_token]) for c in chords]

    # converts a list of indices to chords
    def to_chords(self, indices):
        return [self.idx_to_chord.get(i, self.unknown_token) for i in indices]


# ----------------------------- Main for building and testing the vocabulary -----------------------------

if __name__ == "__main__":
    import sys
    import os
    
    # Add project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.utils.logger import setup_logger
    from src.utils.music_theory import parse_compact_chord, chord_to_roman, roman_to_chord, compact_chord
    from src.config import CLEAN_DATA_PKL, VOCAB_PATH
    
    logger = setup_logger()
    
    # Configuration
    DATA_PATH = CLEAN_DATA_PKL
    
    logger.info("Starting Vocabulary Creation and Testing...")
    
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found at {DATA_PATH}. Run data processing first.")
    else:
        # 1. Load Data & Build Vocab
        logger.info(f"Loading processed songs from {DATA_PATH}...")
        with open(DATA_PATH, 'rb') as f:
            songs = pickle.load(f)
            
        logger.info(f"Loaded {len(songs)} songs.")
        
        vocab = Vocabulary()
        logger.info("Building vocabulary...")
        vocab_size = vocab.build_vocab(songs)
        logger.info(f"Vocabulary built. Size: {vocab_size}")
        
        vocab.save_vocab(VOCAB_PATH)
        logger.info(f"Vocabulary saved to {VOCAB_PATH}")
        
        # 2. Test Vocabulary with Round Trip Conversion
        logger.info("\n--- Testing Vocabulary Round Trip (Text -> Roman -> Idx -> Roman -> Text) ---")
        
        # Test Case: Key of C Major
        test_key = 'C'
        test_chords = [
            "C", "Am", "G7", "F", "Ddim", "Em",
            "E7", "A7", "Dsus2", "Fmaj7", "Bbadd9" 
        ]
        
        logger.info(f"Test Key: {test_key}")
        logger.info(f"{'Original':<8} -> {'Roman':<6} -> {'Idx':<3} -> {'Rec.Roman':<9} -> {'Rec.Chord':<9} [{'Status'}]")
        logger.info("-" * 65)
        
        for chord_str in test_chords:
            # 1. Forward: Text -> Roman -> Index
            root, quality = parse_compact_chord(chord_str)
            roman = chord_to_roman(test_key, root, quality)
            idx = vocab.chord_to_idx.get(roman, vocab.chord_to_idx[vocab.unknown_token])
            
            # 2. Backward: Index -> Roman -> Text
            rec_roman = vocab.idx_to_chord[idx]
            rec_root, rec_quality = roman_to_chord(test_key, rec_roman)
            rec_chord = compact_chord(rec_root, rec_quality)
            
            # 3. Verify
            # We check if the reconstructed roman matches the original roman (Vocabulary consistency)
            # And if the reconstructed chord matches the original chord (Music Theory consistency)
            vocab_match = (roman == rec_roman)
            chord_match = (chord_str == rec_chord)
            
            status = "PASS" if vocab_match and chord_match and idx != vocab.chord_to_idx[vocab.unknown_token] else "FAIL"
            if idx == vocab.chord_to_idx[vocab.unknown_token]: 
                status = "UNK "
            elif vocab_match and not chord_match:
                status = "PART"  # Roman numeral correct but chord differs
            
            logger.info(f"{chord_str:<8} -> {roman:<6} -> {idx:<3} -> {rec_roman:<9} -> {rec_chord:<9} [{status}]")
            
        logger.info("Vocabulary test complete.")