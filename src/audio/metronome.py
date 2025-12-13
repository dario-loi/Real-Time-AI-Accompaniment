# ==================================================================================================================
# Metronome Module
# Plays metronome clicks on each beat with an accent on the first beat of each bar
# ==================================================================================================================

import time
import threading
from typing import Optional
try:
    import fluidsynth
except ImportError:
    fluidsynth = None

from src.config import (
    EMPTY_BARS_COUNT,
    METRONOME_SOUNDFONT, METRONOME_PROGRAM, METRONOME_CHANNEL, METRONOME_GAIN,
    METRONOME_NOTE, METRONOME_VELOCITY, METRONOME_DURATION
)
from src.utils.logger import setup_logger
from src.audio.synth import start_audio_driver, load_soundfont
from src.audio.clock import Conductor

logger = setup_logger()


class Metronome:
    
    def __init__(self, bpm: int, beats_per_bar: float, empty_bars_count: int = EMPTY_BARS_COUNT,
                 soundfont_path: str = METRONOME_SOUNDFONT, 
                 program: int = METRONOME_PROGRAM,
                 channel: int = METRONOME_CHANNEL,
                 gain: float = METRONOME_GAIN,
                 conductor: Optional[Conductor] = None):
        
        self.bpm = bpm
        self.beats_per_bar = beats_per_bar
        self.empty_bars_count = empty_bars_count            # number of empty bars to play before starting the sequence
        self.channel = channel                              # channel to play the metronome on
        self.note = METRONOME_NOTE                          # note to play
        self.velocity = METRONOME_VELOCITY                  # velocity of the note
        self.duration = METRONOME_DURATION                  # duration of the note
        
        self.conductor = conductor                          # conductor to sync with
        self.synth = self._init_synth(soundfont_path, program, channel, gain)

    # Initialize FluidSynth
    def _init_synth(self, soundfont_path, program, channel, gain):
        if fluidsynth is None:
            return None
            
        try:
            # Create synth instance
            synth = fluidsynth.Synth(gain=gain, samplerate=44100.0)
            # Start audio driver (will raise RuntimeError if it fails)
            start_audio_driver(synth, "METRONOME")
            # Load SoundFont using shared helper
            sfid = load_soundfont(synth, soundfont_path)
            # Select metronome sound
            synth.program_select(channel, sfid, 0, program)
            return synth
            
        except Exception as e:
            logger.error(f"[METRONOME] Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Stops and deletes the synth
    def cleanup(self):
        if self.synth:
            try:
                self.synth.delete()
            except Exception as e:
                logger.error(f"[METRONOME] Error during cleanup: {e}")
            self.synth = None

    # Runs the metronome loop. Designed to be run in a separate thread.
    def run(self, start_time_provider, is_running_provider, max_sequence_length):
        """
        Args:
            start_time_provider: Lambda returning the start time (or None)
            is_running_provider: Lambda returning True if the pipeline is running
            max_sequence_length: Maximum number of chords to play
        """
        if not self.synth:
            logger.error("[METRONOME] Synth not initialized")
            return
        
        # Choose between dynamic or static metronome
        if self.conductor:
            self._run_dynamic(is_running_provider)
        else:
            self._run_static(start_time_provider, is_running_provider, max_sequence_length)

    
    # ------------------------------------------ Dynamic BPM player -------------------------------

    def _run_dynamic(self, is_running_provider):
        # Wait for conductor to start
        while not self.conductor.is_running and is_running_provider():
            time.sleep(0.01)
            
        if not is_running_provider():
            return
            
        beat_count = 0
        next_beat_time = self.conductor.start_time  # Sync with conductor's time reference
        
        while is_running_provider():
            current_time = time.time()
            
            # 1. Wait for next beat
            wait_time = next_beat_time - current_time
            
            if wait_time > 0.005:
                time.sleep(wait_time)
                # Precision spin-wait
                while time.time() < next_beat_time:
                    pass

            # Re-sync if verified lag is high
            elif wait_time < -0.1:
                next_beat_time = current_time
            
            # 2. Check if we should stop
            if not is_running_provider():
                break

            # 3. Play click
            beat_duration = self.conductor.beat_duration
            
            # Accent first beat
            velocity_boost = 20 if (beat_count % int(self.beats_per_bar)) == 0 else 0
            click_velocity = self.velocity + velocity_boost
            
            # Play click
            try:
                if self.synth:
                    self.synth.noteon(self.channel, self.note, click_velocity)
                    time.sleep(self.duration)
                    self.synth.noteoff(self.channel, self.note)
            except Exception as e:
                logger.error(f"[METRONOME] Playback error: {e}")
            
            # 4. Schedule next beat
            next_beat_time += beat_duration
            beat_count += 1

    
    # ------------------------------------------ Static BPM player -------------------------------

    def _run_static(self, start_time_provider, is_running_provider, max_sequence_length):
        
        beat_duration = 60.0 / self.bpm
        
        # Wait for start time
        while start_time_provider() is None and is_running_provider():
            time.sleep(0.01)
        
        if not is_running_provider():
            return
        
        start_time_value = start_time_provider()
        delay_beats = int(self.empty_bars_count * self.beats_per_bar)
        total_beats = delay_beats + int(max_sequence_length * self.beats_per_bar)
        
        beat_count = 0
        while is_running_provider() and beat_count < total_beats:
            beat_time = start_time_value + (beat_count * beat_duration)
            current_time = time.time()
            
            wait_time = beat_time - current_time
            if wait_time > 0:
                time.sleep(wait_time)
            
            click_velocity = self.velocity + 20 if beat_count % self.beats_per_bar == 0 else self.velocity
            
            try:
                if self.synth:
                    self.synth.noteon(self.channel, self.note, click_velocity)
                    time.sleep(self.duration)
                    if self.synth:
                        self.synth.noteoff(self.channel, self.note)
            except Exception as e:
                logger.error(f"[METRONOME] Playback error: {e}")
            
            beat_count += 1
        
        logger.info("[METRONOME] Playback complete")


# ------------------------------------------ Test Main ------------------------------------------

if __name__ == "__main__":
    metronome = Metronome(bpm=120, beats_per_bar=4)
    start_time = time.time() + 1
    is_running = True
    
    def get_start_time():
        return start_time
        
    def get_is_running():
        return is_running
        
    t = threading.Thread(target=metronome.run, args=(get_start_time, get_is_running, 4))
    t.start()
    
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        is_running = False
        t.join()
        metronome.cleanup()