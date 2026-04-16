import numpy as np

class RissetGen():
    def __init__(self, sampling_rate=10000):
        self.fs = sampling_rate

    def get_time_axis(self, duration):
        return np.linspace(0, duration, int(self.fs * duration), endpoint=False)

    def gaussian_envelope(self, tempo, t_min, t_max):
        """Gaussian envelope over the log-tempo scale."""
        t_center = np.sqrt(t_min * t_max)
        sigma = np.log2(t_max / t_min) / 4
        dist = np.log2(tempo / t_center)
        return np.exp(-0.5 * (dist / sigma)**2)

    def generate_click_train(self, duration, base_tempo, num_layers=5, direction=1):
        """
        Generates a Risset Rhythm (accelerando/decelerando).
        base_tempo: starting BPM for the lowest layer.
        direction: 1 for accelerating, -1 for decelerating.
        """
        t = self.get_time_axis(duration)
        final_signal = np.zeros_like(t)
        
        # We'll use BPM as the tempo unit
        # base_tempo is the minimum tempo of the cycle
        t_min = base_tempo
        t_max = base_tempo * (2**num_layers)
        
        for i in range(num_layers):
            # Each layer's relative tempo
            layer_start_tempo = base_tempo * (2**i)
            
            # Instantaneous tempo in BPM
            # log_tempo = (log_start + dir * t/dur) % num_layers
            log_start = np.log2(layer_start_tempo / t_min)
            rel_t = (log_start + direction * t / duration) % num_layers
            curr_tempo_bpm = t_min * (2**rel_t)
            curr_tempo_hz = curr_tempo_bpm / 60.0
            
            # To generate clicks at an accelerating rate, we need to track phase of the "beat"
            # Phase = Integral(tempo_hz dt)
            # phase(t) = (t_min/60) * (duration / (direction * ln(2))) * (2^rel_t)
            phase = (t_min / 60.0) * (duration / (direction * np.log(2))) * (2**rel_t)
            
            # Use sine of phase to generate a 'click' pulse
            # We want sharp pulses, so we can take sin(phase)^high_power or similar
            # A pulse whenever phase passes a multiple of 2*pi
            pulses = np.sin(2 * np.pi * phase)
            # Sharpen pulses:
            pulses = np.where(pulses > 0.98, 1.0, 0.0)
            
            # Apply envelope
            envelope = self.gaussian_envelope(curr_tempo_bpm, t_min, t_max)
            final_signal += pulses * envelope
            
        # Normalize
        if np.max(np.abs(final_signal)) > 0:
            final_signal /= np.max(np.abs(final_signal))
            
        return final_signal

if __name__ == "__main__":
    gen = RissetGen()
    # 60 BPM base, 5 layers, 10 seconds
    rhythm = gen.generate_click_train(duration=10, base_tempo=60)
    print("Risset Rhythm generated.")
