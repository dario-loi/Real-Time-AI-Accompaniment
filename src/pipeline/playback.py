# ===============================================================================
#                                      PlaybackThread
# ===============================================================================
# Plays chords in sequence with timing alignment
# ===============================================================================

import time
import threading
from src.audio.midi_io import play_chord
from src.utils.logger import setup_logger

logger = setup_logger()

class PlaybackThread(threading.Thread):
    def __init__(self, chord_objects, start_time_func, delay_seconds, chord_duration_seconds, output_port, max_sequence_length, is_running_func):
        super().__init__(daemon=True)
        self.chord_objects = chord_objects                          # Shared list containting the chords to play
        self.get_start_time = start_time_func                       # Function to get the start time
        self.delay_seconds = delay_seconds                          # Delay before starting the sequence
        self.chord_duration_seconds = chord_duration_seconds        # Duration of each chord
        self.output_port = output_port                              # MIDI output port
        self.max_sequence_length = max_sequence_length              # Maximum chords to play
        self.is_running_func = is_running_func                      # Function to check if the pipeline is running
        self.is_running_internal = False                            # Internal flag to control thread
        
    def run(self):
        self.is_running_internal = True
        last_played_idx = -1                                        # Index of last played chord
        
        # Run while internal flag is True AND pipeline is running
        while self.is_running_internal and self.is_running_func():
            # Check if new chords are available
            if len(self.chord_objects) > last_played_idx + 1:
                chord_to_play = self.chord_objects[last_played_idx + 1]
                
                # Timing
                chord_idx = last_played_idx + 1
                start_time = self.get_start_time()
                if start_time is None:
                    time.sleep(0.1)
                    continue
                
                # Calculate play time for alignment
                play_time = start_time + self.delay_seconds + (chord_idx * self.chord_duration_seconds)
                
                wait_time = play_time - time.time()
                if wait_time > 0:
                    time.sleep(wait_time)
                    
                # Play
                if self.output_port:
                    play_chord(chord_to_play, self.output_port)
                
                last_played_idx += 1
                
                # Auto-terminate if sequence is complete
                if last_played_idx + 1 >= self.max_sequence_length:
                    time.sleep(self.chord_duration_seconds)
                    self.is_running_internal = False
                    break
                    
            else:
                time.sleep(0.01)
                
    def stop(self):
        self.is_running_internal = False
