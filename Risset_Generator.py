import numpy as np

class RissetGen():
    def __init__(self, sampling_rate=10000):
        self.fs = sampling_rate

    def get_time_axis(self, duration):
        return np.linspace(0, duration, int(self.fs * duration), endpoint=False)

    def hann_envelope(self, tempo, t_min, t_max):
        """Hann window envelope over the log-tempo scale. Fully tapers to 0."""
        log_t = np.log2(tempo)
        log_min = np.log2(t_min)
        log_max = np.log2(t_max)
        
        # Normalize between 0 and 1
        normalized_pos = (log_t - log_min) / (log_max - log_min)
        
        # Apply Hann window formula: 0.5 * (1 - cos(2 * pi * x))
        envelope = np.where((normalized_pos >= 0) & (normalized_pos <= 1),
                            0.5 * (1 - np.cos(2 * np.pi * normalized_pos)), 
                            0.0)
        return envelope

    def generate_click_train(self, duration, base_tempo, num_layers=5, direction=1, active_layers=None, use_envelope=True):
        """
        Generates a Risset Rhythm (accelerando/decelerando).
        Phase continuity is maintained to avoid glitches.
        """
        t = self.get_time_axis(duration)
        final_signal = np.zeros_like(t)
        
        # Tempo boundaries (BPM)
        t_min = base_tempo
        t_max = base_tempo * (2**num_layers)
        
        k = direction / duration
        
        for i in range(num_layers):
            if active_layers is not None and i not in active_layers:
                continue
                
            # Starting tempo for this layer
            tempo0 = base_tempo * (2**i)
            
            # Wrapped tempo for envelope
            log_start = np.log2(tempo0 / t_min)
            rel_t = (log_start + direction * t / duration) % num_layers
            curr_tempo_bpm = t_min * (2**rel_t)
            
            # Beat phase: phase(t) = integral(tempo_hz dt)
            # tempo_hz = (tempo0 / 60) * 2^(k*t)
            if k == 0:
                phase = (tempo0 / 60.0) * t
            else:
                phase = (tempo0 / 60.0) * (2**(k * t) - 1) / (k * np.log(2))
            
            # Use sine of phase to generate a 'click' pulse
            pulses = np.sin(2 * np.pi * phase)
            # Sharpen pulses:
            pulses = np.where(pulses > 0.98, 1.0, 0.0)
            
            # Apply envelope
            if use_envelope:
                envelope = self.hann_envelope(curr_tempo_bpm, t_min, t_max)
                final_signal += pulses * envelope
            else:
                final_signal += pulses
            
        # Normalize
        if np.max(np.abs(final_signal)) > 0:
            final_signal /= np.max(np.abs(final_signal))
            
        final_signal[-1] = 0
        np.savetxt("risset_rhythm.txt", final_signal, delimiter="\n")
        return final_signal
            
        # Normalize
        if np.max(np.abs(final_signal)) > 0:
            final_signal /= np.max(np.abs(final_signal))
            
        final_signal[-1] = 0
        np.savetxt("risset_rhythm.txt", final_signal, delimiter="\n")
        return final_signal

if __name__ == "__main__":
    gen = RissetGen()
    # 60 BPM base, 5 layers, 10 seconds
    rhythm = gen.generate_click_train(duration=10, base_tempo=60)
    print("Risset Rhythm generated.")
