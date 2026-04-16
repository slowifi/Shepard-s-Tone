import numpy as np

class SigGen():
    def __init__(self, sampling_rate=10000):
        self.fs = sampling_rate

    def get_time_axis(self, duration):
        return np.linspace(0, duration, int(self.fs * duration), endpoint=False)

    def gaussian_envelope(self, freq, f_min, f_max):
        """Gaussian envelope to fade tones in and out at frequency boundaries."""
        f_center = np.sqrt(f_min * f_max)
        sigma = np.log2(f_max / f_min) / 4
        dist = np.log2(freq / f_center)
        return np.exp(-0.5 * (dist / sigma)**2)

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

    def generate_shepard_tone(self, duration=10, base_f=100, num_layers=5, direction=1):
        """
        Generates a Shepard Tone by summing multiple frequency-swept layers.
        Each layer is separated by one octave.
        """
        t = self.get_time_axis(duration)
        final_signal = np.zeros_like(t)
        
        f_min = base_f
        f_max = base_f * (2**num_layers)
        
        for i in range(num_layers):
            # Each layer starts at a different octave
            f0 = base_f * (2**i)
            
            # Instantaneous frequency: f(t) = f0 * 2^(t/duration)
            # To wrap around: we use modulo in the log2 space
            # log2(f/f_min) = (log2(f0/f_min) + t/duration) % num_layers
            log_f0 = np.log2(f0 / f_min)
            rel_t = (log_f0 + direction * t / duration) % num_layers
            curr_freq = f_min * (2**rel_t)
            
            # Integral of f_min * 2^((log_f0 + t/dur) % num_layers)
            # This is tricky due to the modulo jump. 
            # Simplified approach: Sum continuous swept sines and fade them.
            
            phase = 2 * np.pi * f_min * (duration / (direction * np.log(2))) * (2**rel_t)
            # The jump in phase at modulo needs to be handled if we want perfect continuity,
            # but for a continuous loop, the phase at t=duration should match t=0.
            
            layer_signal = np.sin(phase)
            
            # Envelope based on the relative position in octaves
            envelope = self.gaussian_envelope(curr_freq, f_min, f_max)
            final_signal += layer_signal * envelope
            
        # Normalize
        if np.max(np.abs(final_signal)) > 0:
            final_signal /= np.max(np.abs(final_signal))
            
        np.savetxt("shepard_tone.txt", final_signal, delimiter="\n")
        return final_signal
