# =============================================================================
# Handles MIDI file generation and playback for Chord objects
# as MIDI events on a specified MIDI output port.
# =============================================================================

import mido
import time
from src.utils.logger import setup_logger

logger = setup_logger()

def save_chords_to_midi(chord_sequence, filename='output.mid', bpm=120):
    """Generates a MIDI file from a sequence of Chord objects."""
    midi_file = mido.MidiFile()
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    
    # Set Tempo
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))
    ticks_per_beat = midi_file.ticks_per_beat
    
    abs_time = 0
    
    for chord in chord_sequence:
        for relative_time, msg in chord.midi_messages:
             # Convert seconds to ticks
             ticks = int(relative_time * ticks_per_beat * bpm / 60)
             
             new_msg = msg.copy()
             new_msg.time = max(0, ticks - abs_time)
             abs_time = ticks
             track.append(new_msg)
             
        # Advance absolute time for next chord
        abs_time += int(chord.beats_per_bar * ticks_per_beat)
        
    midi_file.save(filename)
    logger.info(f"MIDI saved: {filename}")

def play_chord(chord, output_port_name):
    """Plays a single Chord object on the specified MIDI port."""
    try:
        with mido.open_output(output_port_name) as out:
            start_time = time.time()
            for relative_time, msg in chord.midi_messages:
                target_time = start_time + relative_time
                current_time = time.time()
                wait_time = target_time - current_time
                if wait_time > 0:
                    time.sleep(wait_time)
                out.send(msg)
    except Exception as e:
        logger.error(f"Playback Error: {e}")

def play_chord_sequence(sequence, output_port_name):
    """Plays a sequence of Chord objects."""
    for i, chord in enumerate(sequence):
        logger.info(f"Playing chord {i+1}/{len(sequence)}: {chord}")
        play_chord(chord, output_port_name)
