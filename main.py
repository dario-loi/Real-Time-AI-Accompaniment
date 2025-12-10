# ==================================================================================================================
# Real-Time Accompaniment Generation Pipeline entry point
# ==================================================================================================================

from src.pipeline import RealTimePipeline
from src.config import OUTPUT_PORT, INPUT_PORT, WINDOW_SIZE
from src.utils.logger import setup_logger
from src.utils.music_theory import get_starting_chord

# Config
BPM = 100
DEFAULT_KEY = 'C'
BEATS_PER_BAR = 4.0
SEQUENCE_LENGTH = 10


logger = setup_logger()

if __name__ == "__main__":
    
    logger.info("=" * 50)
    logger.info("REAL-TIME ACCOMPANIMENT PIPELINE")
    logger.info("=" * 50)
    
    # Interactive Setup
    system_key, start_root, start_quality = get_starting_chord(DEFAULT_KEY)
    
    logger.info(f"Starting System: Key={system_key} | StartChord={start_root}{start_quality}")
    
    try:
        pipeline = RealTimePipeline(
            key=system_key,                             # Musical key (for AI context)
            starting_root=start_root,                   # Actual starting chord root
            starting_quality=start_quality,             # Actual starting chord quality
            bpm=BPM,                                    # Tempo in BPM
            beats_per_bar=BEATS_PER_BAR,                # How many beats in one bar
            window_size=WINDOW_SIZE,                    # How many chords to consider for prediction
            max_sequence_length=SEQUENCE_LENGTH,        # Maximum generated sequence length in chords
            output_port=OUTPUT_PORT,                    # Output port for MIDI playback (listen)
            input_port=INPUT_PORT,                      # Input port for MIDI input (play)
            enable_input_listener=True,                 # Enable MIDI input listener
            enable_metronome=True,                      # Enable metronome
            enable_synth=True                           # Enable MIDI synth
        )
        
        # Start pipeline
        final_sequence = pipeline.start()
        
        # Show results
        logger.info("=" * 50)
        logger.info("FINAL SEQUENCE:")
        logger.info("=" * 50)
        
        sequence_names = pipeline.get_current_sequence()
        for i, (chord_obj, chord_name) in enumerate(zip(final_sequence, sequence_names)):
            logger.info(f"Chord {i+1}: {chord_name} ({chord_obj.root} {chord_obj.chord_type})")
        
        logger.info(f"Total duration: {len(final_sequence) * pipeline.chord_duration_seconds:.1f} seconds")
        
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()