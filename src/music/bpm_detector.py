# ==============================================================================
# BPM Detector - Real-time tempo detection using Inter-Onset Intervals (IOI)
# ==============================================================================
# Detects BPM from note onsets using a sliding window of IOIs.
# Features:
#   - Hysteresis to prevent jitter (DELTA_RETAIN_BPM threshold)
#   - Tempo jump detection (TEMPO_JUMP_THRESHOLD)
# ==============================================================================

import time
import threading
from collections import deque
from typing import Optional, List
from statistics import median

from src.config import (
    BPM_MIN, BPM_MAX,
    IOI_MIN, IOI_MAX, MINIMUM_IOI_NUMBER, IOI_WINDOW_SIZE,
    TEMPO_JUMP_THRESHOLD, CONSECUTIVE_JUMP_THRESHOLD, DELTA_RETAIN_BPM
)
from src.audio.clock import Conductor
from src.utils.logger import setup_logger

logger = setup_logger()


class BPMDetector:
    """
    Checks for tempo jumps BEFORE adding IOI to buffer,
    ensuring jumps are detected before the buffer becomes mixed.
    """
    
    def __init__(self, conductor: Optional[Conductor] = None):
        self._conductor = conductor
        
        self._ioi_buffer: deque = deque(maxlen=IOI_WINDOW_SIZE)                    # IOI sliding window buffer
        self._last_onset: Optional[float] = None                                   # Last onset timestamp
        self._current_bpm: Optional[float] = conductor.bpm if conductor else None  # Current BPM estimate
        
        # Jump detection: track consecutive outlier IOIs
        self._consecutive_jumps = 0
        self._jump_candidate_bpm: Optional[float] = None

        # Thread safety
        self._lock = threading.RLock()
    
    # ----------------------------------- Properties -----------------------------------
    
    @property
    def bpm(self) -> Optional[float]:
        with self._lock:
            return self._current_bpm
    
    # ----------------------------------- Core Functions -----------------------------------
    
    def add_onset(self, timestamp: float) -> Optional[float]:
        """
        Args:
            timestamp: Time of the note onset (time.time())
        
        Returns:
            Current BPM estimate (or None if not yet available)
        """
        with self._lock:
            if self._last_onset is not None:
                ioi = timestamp - self._last_onset
                
                # Convert IOI to instant BPM (clamped to valid range)
                instant_bpm = 60.0 / ioi if ioi > 0 else 0
                instant_bpm = max(BPM_MIN, min(BPM_MAX, instant_bpm))
                
                # Validity checks
                ioi_in_range = IOI_MIN <= ioi <= IOI_MAX
                
                # DELTA_RETAIN_BPM% change threshold (ignore tiny fluctuations)
                significant_change = True
                if self._current_bpm is not None and ioi_in_range:
                    percent_change = abs(instant_bpm - self._current_bpm) / self._current_bpm
                    significant_change = percent_change >= DELTA_RETAIN_BPM
                
                valid = ioi_in_range and significant_change
                logger.info(f"[BPM] IOI: {ioi:.3f}s → {instant_bpm:.1f} BPM (valid: {valid})")
                
                # Only process valid IOIs
                if ioi_in_range:
                    # Check for tempo jump BEFORE adding to buffer
                    if self._check_tempo_jump(ioi):
                        pass  # Jump detected and handled
                    else:
                        # Normal update
                        self._ioi_buffer.append(ioi)
                        self._update_bpm()
            
            self._last_onset = timestamp
            return self._current_bpm
    
    def _check_tempo_jump(self, new_ioi: float) -> bool:
        """
        Check if this IOI represents a tempo jump (e.g., player suddenly doubles/halves tempo).
        
        Returns:
            True if jump was triggered (caller should skip normal update)
        """
        if self._current_bpm is None:
            return False
        
        instant_bpm = 60.0 / new_ioi
        instant_bpm = max(BPM_MIN, min(BPM_MAX, instant_bpm))
        
        # Check if this IOI is an outlier (> TEMPO_JUMP_THRESHOLD% change)
        percent_change = abs(instant_bpm - self._current_bpm) / self._current_bpm
        is_jump_ioi = percent_change >= TEMPO_JUMP_THRESHOLD
        
        if is_jump_ioi:
            self._consecutive_jumps += 1
            
            # Track candidate BPM for the jump
            if self._jump_candidate_bpm is None:
                self._jump_candidate_bpm = instant_bpm
            else:
                self._jump_candidate_bpm = (self._jump_candidate_bpm + instant_bpm) / 2
            
            # Need multiple consecutive outliers to confirm a tempo jump
            if self._consecutive_jumps >= CONSECUTIVE_JUMP_THRESHOLD:
                self._trigger_tempo_jump(self._current_bpm, self._jump_candidate_bpm)
                return True
        else:
            # Reset jump tracking if we get a normal IOI
            self._consecutive_jumps = 0
            self._jump_candidate_bpm = None
        
        return False
    
    def _trigger_tempo_jump(self, old_bpm: float, new_bpm: float):
        """Handle a confirmed tempo jump by resetting state."""
        logger.info(f"[BPM] ⚡ TEMPO JUMP: {old_bpm:.1f} → {new_bpm:.1f} BPM")
        
        # Clear buffer and reset with new BPM
        self._ioi_buffer.clear()
        self._current_bpm = new_bpm
        self._consecutive_jumps = 0
        self._jump_candidate_bpm = None
        
        if self._conductor:
            self._conductor.force_tempo(new_bpm)
    
    def clear(self):
        """Clear IOI buffer (e.g., after a pause)."""
        with self._lock:
            old_bpm = self._current_bpm
            self._ioi_buffer.clear()
            self._last_onset = None
            self._consecutive_jumps = 0
            self._jump_candidate_bpm = None
            logger.info(f"[BPM] Buffer cleared" + (f" (was {old_bpm:.1f} BPM)" if old_bpm else ""))
    
    def reset(self):
        """Full reset including BPM estimate."""
        with self._lock:
            self.clear()
            self._current_bpm = None
    
    # ------------------------------- Internal Functions -------------------------------
    
    # BPM update logic: use mean of IOI buffer
    def _update_bpm(self):
        # Need minimum samples before calculating BPM
        if len(self._ioi_buffer) < MINIMUM_IOI_NUMBER:
            return
        
        # Calculate mean BPM from buffer
        iois = list(self._ioi_buffer)
        mean_ioi = sum(iois) / len(iois)
        new_bpm = 60.0 / mean_ioi
        new_bpm = max(BPM_MIN, min(BPM_MAX, new_bpm))
        
        old_bpm = self._current_bpm
        
        # HYSTERESIS: Only update if change > DELTA_RETAIN_BPM%
        percent_change = abs(new_bpm - old_bpm) / old_bpm
        if percent_change < DELTA_RETAIN_BPM:
            return
        
        # Apply update
        self._current_bpm = new_bpm
        
        start_time = self._conductor._start_time if self._conductor else 0
        elapsed = time.time() - start_time if start_time else 0
        logger.info(f"[{elapsed:.1f}s] [BPM] Update: {old_bpm:.1f} -> {new_bpm:.1f} BPM")
        
        if self._conductor:
            self._conductor.force_tempo(new_bpm)
    
    def get_stats(self) -> dict:
        """Get detector statistics for debugging."""
        with self._lock:
            iois = list(self._ioi_buffer)
            return {
                'bpm': self._current_bpm,
                'buffer_size': len(self._ioi_buffer),
                'ioi_mean': sum(iois) / len(iois) if iois else None,
                'ioi_median': median(iois) if iois else None,
                'consecutive_jumps': self._consecutive_jumps,
            }
    
    def __repr__(self) -> str:
        bpm_str = f"{self._current_bpm:.1f}" if self._current_bpm else "?"
        return f"BPMDetector({bpm_str} BPM)"
