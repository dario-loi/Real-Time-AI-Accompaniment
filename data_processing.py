# =================================================================
# Data Processing Module
#
# Processes raw Chordonomicon dataset and turns it into a clean
# roman numeral dataset ready for training.
# It does so by using a custom keydetector to detect the key of each
# song with a confidence threshold of 0.79.
# =================================================================

import pandas as pd
import pickle
import multiprocessing
from tqdm import tqdm
from typing import List, Optional, Tuple, Dict

from src.music.key_detector_major import KeyDetector
from src.utils.music_theory import chord_to_roman, parse_compact_chord
from src.config import NOTE_TO_MIDI_MAP, INTERVALS_MAP, NOTES, RAW_DATA_PATH, CLEAN_DATA_PKL, CLEAN_DATA_CSV
from src.utils.logger import setup_logger

logger = setup_logger()

# Cleans raw chord string from Chordonomicon dataset
def clean_chord(chord_str: str) -> Optional[str]:
    if chord_str.startswith('<'): return None
    
    # Protect 'sus' from 's' -> '#' replacement
    chord_str = chord_str.replace('sus', 'SUS')
    chord_str = chord_str.replace('s', '#')     # s is used for #
    chord_str = chord_str.replace('SUS', 'sus')
    
    # Remove slash chords (keep root only)
    if '/' in chord_str:
        chord_str = chord_str.split('/')[0]
        
    # Remove specific suffixes
    for suffix in ['no3d', 'no5', '(b5)']:
        chord_str = chord_str.replace(suffix, '')
    
    return chord_str.replace('min', 'm')

# Converts list of chord strings to list of MIDI note numbers
def get_notes_from_chords(chord_list: List[str]) -> List[int]:
    all_notes = []
    for chord_str in chord_list:
        root, quality = parse_compact_chord(chord_str)
        
        if root not in NOTE_TO_MIDI_MAP: 
            logger.warning(f"Invalid root note: {root}")
            continue
            
        root_midi = NOTE_TO_MIDI_MAP[root]
        intervals = INTERVALS_MAP.get(quality, INTERVALS_MAP['major'])
        
        all_notes.extend([(root_midi + i) % 12 for i in intervals])
        
    return all_notes

# Processes a single song row returning (numerals_list, inspection_dict) or None
def process_row(row_data: Tuple) -> Optional[Tuple[List[str], Dict]]:

    row_id, raw_chords_str, artist_id = row_data
    
    if not isinstance(raw_chords_str, str): 
        logger.warning(f"Invalid chords string: {raw_chords_str}")
        return None
    
    # Clean chords
    clean_chords = [clean_chord(t) for t in raw_chords_str.split()]
    clean_chords = [c for c in clean_chords if c] # Filter None
    
    if not clean_chords: return None

    # 1. Detect Key
    song_notes = get_notes_from_chords(clean_chords)
    if not song_notes: 
        logger.warning(f"Invalid song notes: {song_notes}")
        return None
    
    # instantiate key detector per process
    key, confidence = KeyDetector(min_notes=3).detect(song_notes) or (None, 0)
    if not key: 
        unique_pcs = sorted(set(song_notes))
        unique_notes = [NOTES[pc] for pc in unique_pcs]
        unique_chords = list(dict.fromkeys(clean_chords))[:8]
        logger.warning(f"\n[FAILED] ID: {row_id}")
        logger.warning(f"  Chords: {' '.join(unique_chords)}{'...' if len(set(clean_chords)) > 8 else ''}")
        logger.warning(f"  Unique notes ({len(unique_notes)}): {', '.join(unique_notes)}")
        return None
    
    # Filter by confidence threshold
    if confidence < 0.79:
        return None
    
    # 2. Convert to Roman Numerals
    numerals = []
    for chord in clean_chords:
        root, quality = parse_compact_chord(chord)
        numerals.append(chord_to_roman(key, root, quality))
        
    inspection_entry = {
        'id': row_id,
        'artist': artist_id if pd.notna(artist_id) else 'Unknown',
        'key': key,
        'confidence': f"{confidence:.2f}",
        'numerals': ' '.join(numerals),
        'original_chords': ' '.join(clean_chords)
    }
    
    return numerals, inspection_entry

# Main function to process entire dataset in parallel
def process_dataset():
    logger.info(f"Loading dataset from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    
    # Prepare tasks (ID, Chords, Artist)
    tasks = [
        (row['id'], row['chords'], row.get('artist_id')) 
        for _, row in df.iterrows()
    ]
    
    processed_songs = []
    inspection_data = []
    
    # Multiprocessing
    num_cores = multiprocessing.cpu_count()
    logger.info(f"Processing {len(tasks)} songs using {num_cores} cores...")
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(process_row, tasks, chunksize=100), total=len(tasks)))
        
    # Aggregate results
    for res in results:
        if res:
            nums, insp = res
            processed_songs.append(nums)
            inspection_data.append(insp)

    # Save Artifacts
    logger.info(f"Saving {len(processed_songs)} songs to {CLEAN_DATA_PKL}...")
    with open(CLEAN_DATA_PKL, 'wb') as f:
        pickle.dump(processed_songs, f)
        
    logger.info(f"Saving inspection data to {CLEAN_DATA_CSV}...")
    inspection_df = pd.DataFrame(inspection_data)
    inspection_df.to_csv(CLEAN_DATA_CSV, index=False)
    
    if not inspection_df.empty:
        logger.info("\nSample Output:")
        logger.info(inspection_df.head().to_string())
    
    logger.info("Done.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    process_dataset()