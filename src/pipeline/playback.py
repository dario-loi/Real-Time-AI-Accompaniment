# ===============================================================================
#                                      PlaybackThread
# ===============================================================================
# Plays chords in sequence with timing alignment.
# Supports both static timing and dynamic timing via Conductor.
# ===============================================================================

import time
import threading
from typing import Optional
from src.audio.midi_io import play_chord
from src.utils.logger import setup_logger
from src.audio.clock import Conductor

logger = setup_logger()

class PlaybackThread(threading.Thread):
    def __init__(self, 
                 chord_objects: list, 
                 output_port: str, 
                 max_sequence_length: int, 
                 is_running_func,
                 conductor: Optional[Conductor] = None,
                 start_time_func = None,
                 delay_seconds: float = 0.0,
                 chord_duration_seconds: float = 0.0,
                 start_beat_offset: float = 0.0,
                 beats_per_bar: float = 4.0):
        
        super().__init__(daemon=True)
        self.chord_objects = chord_objects
        self.output_port = output_port
        self.max_sequence_length = max_sequence_length
        self.is_running_func = is_running_func
        
        # Dynamic timing
        self.conductor = conductor
        self.start_beat_offset = start_beat_offset
        self.beats_per_bar = beats_per_bar
        
        # Static timing
        self.get_start_time = start_time_func
        self.delay_seconds = delay_seconds
        self.chord_duration_seconds = chord_duration_seconds
        
        self.is_running_internal = False
        
    def run(self):
        self.is_running_internal = True
        last_played_idx = -1
        
        # choose between dynamic or static timing
        if self.conductor:
            self._run_dynamic(last_played_idx)
        else:
            self._run_static(last_played_idx)

    def _run_dynamic(self, last_played_idx):
        """Run with Conductor - plays immediately when chord is added."""
        # Wait for conductor start
        while not self.conductor.is_running and self.is_running_func():
            time.sleep(0.01)

        while self.is_running_internal and self.is_running_func():
            # Check for new chords - play immediately when one appears
            if len(self.chord_objects) > last_played_idx + 1:
                chord_idx = last_played_idx + 1
                chord_to_play = self.chord_objects[chord_idx]
                
                # Play immediately - timing is handled by _timing_thread
                if self.output_port:
                    play_chord(chord_to_play, self.output_port)
                
                last_played_idx += 1
                
                # End sequence check
                if last_played_idx + 1 >= self.max_sequence_length:
                    # Wait for duration of last chord before ending
                    time.sleep(self.conductor.bar_duration)
                    self.is_running_internal = False
                    break
            else:
                # Poll for new chords
                time.sleep(0.005)  # 5ms poll - fast response

    def _run_static(self, last_played_idx):
        """Run with static timing (legacy)."""
        while self.is_running_internal and self.is_running_func():
            if len(self.chord_objects) > last_played_idx + 1:
                chord_to_play = self.chord_objects[last_played_idx + 1]
                chord_idx = last_played_idx + 1
                
                start_time = self.get_start_time()
                if start_time is None:
                    time.sleep(0.1)
                    continue
                
                play_time = start_time + self.delay_seconds + (chord_idx * self.chord_duration_seconds)
                wait_time = play_time - time.time()
                
                if wait_time > 0:
                    time.sleep(wait_time)
                    
                if self.output_port:
                    play_chord(chord_to_play, self.output_port)
                
                last_played_idx += 1
                
                if last_played_idx + 1 >= self.max_sequence_length:
                    time.sleep(self.chord_duration_seconds)
                    self.is_running_internal = False
                    break
            else:
                time.sleep(0.01)
                
    def stop(self):
        self.is_running_internal = False
