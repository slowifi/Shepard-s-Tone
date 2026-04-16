import nidaqmx
from nidaqmx.constants import AcquisitionType, RegenerationMode
import numpy as np
from Signal_Generator import SigGen

class DAQOutput:
    def __init__(self, device_name="Dev1", channel="ao0", sampling_rate=10000):
        self.device_name = device_name
        self.channel = f"{device_name}/{channel}"
        self.fs = sampling_rate

    def send_signal_continuous(self, signal):
        """
        Sends a signal to the DAQ and loops it continuously until the user presses Enter.
        """
        try:
            with nidaqmx.Task() as task:
                task.ao_channels.add_ao_voltage_chan(self.channel)
                
                # Configure timing for continuous output
                task.timing.cfg_samp_clk_timing(
                    rate=self.fs,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=len(signal)
                )
                
                # Allow regeneration so the buffer loops automatically
                task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
                
                # Write initial data to buffer
                task.write(signal, auto_start=False)
                
                print(f"Starting continuous output on {self.channel}...")
                print("The signal will loop indefinitely.")
                
                task.start()
                
                input("\n>>> Vibration active. Press [ENTER] to stop <<<\n")
                
                task.stop()
                print("Vibration stopped.")
                
        except Exception as e:
            print(f"Error outputting to DAQ: {e}")

    def send_signal_finite(self, signal):
        """
        Sends a signal once (finite mode).
        """
        try:
            with nidaqmx.Task() as task:
                task.ao_channels.add_ao_voltage_chan(self.channel)
                task.timing.cfg_samp_clk_timing(
                    rate=self.fs,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=len(signal)
                )
                
                print(f"Sending finite signal to {self.channel}...")
                task.write(signal, auto_start=True)
                task.wait_until_done(timeout=len(signal)/self.fs + 2)
                print("Signal output complete.")
        except Exception as e:
            print(f"Error outputting to DAQ: {e}")

if __name__ == "__main__":
    fs = 10000
    gen = SigGen(sampling_rate=fs)

    # Generate a short segment to loop (e.g., 2 seconds)
    shepard_signal = gen.generate_shepard_tone(duration=5)

    daq = DAQOutput(sampling_rate=fs)
    

    daq.send_signal_finite(shepard_signal)
    # daq.send_signal_continuous(shepard_signal)
