import numpy as np

class SigGen():
    def __init__(self, sampling_rate=10000):
        self.fs = sampling_rate

    def get_time_axis(self, duration):
        return np.linspace(0, duration, int(self.fs * duration), endpoint=False)

    def hann_envelope(self, freq, f_min, f_max):
        """Hann window envelope to fade tones in and out at frequency boundaries. Fully tapers to 0."""
        # Convert frequencies to log scale to apply symmetric window
        log_f = np.log2(freq)
        log_min = np.log2(f_min)
        log_max = np.log2(f_max)
        
        # Normalize between 0 and 1
        normalized_pos = (log_f - log_min) / (log_max - log_min)
        
        # Apply Hann window formula: 0.5 * (1 - cos(2 * pi * x))
        # Valid only in [0, 1] range, 0 otherwise
        envelope = np.where((normalized_pos >= 0) & (normalized_pos <= 1),
                            0.5 * (1 - np.cos(2 * np.pi * normalized_pos)), 
                            0.0)
        return envelope

    def generate_shepard_layer(self, t, initial_freq, duration, direction=1):
        """Generates a single frequency-swept sine wave with its volume enveloped."""
        # Logarithmic frequency sweep: f(t) = f0 * 2^(direction * t / duration)
        # Phase is the integral of frequency: phi(t) = 2*pi * integral(f(t) dt)
        # phi(t) = 2*pi * f0 * duration / (direction * ln(2)) * (2^(direction * t / duration) - 1)
        
        f_min = initial_freq / 2
        f_max = initial_freq * (2**5) # 5 octaves range example
        
        # Calculate instantaneous frequency and phase
        # Note: direction +1 for ascending, -1 for descending
        freq_t = initial_freq * 2**(direction * t / duration)
        phase = 2 * np.pi * initial_freq * (duration / (direction * np.log(2))) * (2**(direction * t / duration) - 1)
        
        signal = np.sin(phase)
        
        # Apply Gaussian envelope based on instantaneous frequency
        # We need to wrap the frequency within the spectral window
        # For simplicity in Shepard Tone, we usually use multiple layers with fixed offsets
        return signal, freq_t

    def generate_shepard_tone(self, duration=10, base_f=100, num_layers=5, direction=1, active_layers=None, use_envelope=True):
        """
        Generates a Shepard Tone by summing multiple frequency-swept layers.
        Phase continuity is maintained by avoiding modulo inside the phase term.
        """
        t = self.get_time_axis(duration)
        final_signal = np.zeros_like(t)
        
        # Spectral window boundaries for the envelope
        f_min = base_f
        f_max = base_f * (2**num_layers)
        
        # Constant for logarithmic sweep: f(t) = f0 * 2^(k*t)
        k = direction / duration
        
        for i in range(num_layers):
            if active_layers is not None and i not in active_layers:
                continue
            
            # Starting frequency for this layer (octave offset)
            f0 = base_f * (2**i)
            
            # Instantaneous frequency: f(t) = f0 * 2^(k*t)
            # To maintain circularity, we wrap the frequency for the envelope calculation
            log_f0 = np.log2(f0 / f_min)
            rel_t = (log_f0 + direction * t / duration) % num_layers
            curr_freq = f_min * (2**rel_t)
            
            # Continuous phase: phi(t) = 2*pi * integral(f(t) dt)
            # Here f(t) = f0 * 2^(k*t)
            # phi(t) = 2*pi * f0 * (2^(k*t) - 1) / (k * ln(2))
            if k == 0:
                phase = 2 * np.pi * f0 * t
            else:
                phase = 2 * np.pi * f0 * (2**(k * t) - 1) / (k * np.log(2))
            
            layer_signal = np.sin(phase)
            
            # Envelope based on the wrapped instantaneous frequency
            if use_envelope:
                envelope = self.hann_envelope(curr_freq, f_min, f_max)
                final_signal += layer_signal * envelope
            else:
                final_signal += layer_signal
            
        # Normalize
        if np.max(np.abs(final_signal)) > 0:
            final_signal /= np.max(np.abs(final_signal))
            
        final_signal[-1] = 0
        np.savetxt("shepard_tone.txt", final_signal, delimiter="\n")
        return final_signal
