import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

# 기존 클래스 임포트
from Signal_Generator import SigGen
from Risset_Generator import RissetGen
from DAQ import DAQOutput

class ShepardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shepard-Risset Haptic Controller (High Visibility)")
        self.root.geometry("1500x1200")

        # 폰트 설정 (기존 대비 약 2배)
        self.default_font = ("Arial", 14)
        self.header_font = ("Arial", 18, "bold")
        
        style = ttk.Style()
        style.configure(".", font=self.default_font)
        style.configure("TButton", font=self.default_font, padding=10)
        style.configure("TLabel", font=self.default_font)
        style.configure("TEntry", font=self.default_font)
        style.configure("TLabelframe.Label", font=self.header_font)

        # 인스턴스 초기화
        self.fs_var = tk.StringVar(value="10000")
        self.sig_gen = SigGen(sampling_rate=int(self.fs_var.get()))
        self.risset_gen = RissetGen(sampling_rate=int(self.fs_var.get()))
        self.daq = None
        
        self.current_signal = None
        self.shepard_layer_vars = []
        self.risset_layer_vars = []

        # Feature Toggles
        self.use_envelope = tk.BooleanVar(value=True)
        self.use_compensation = tk.BooleanVar(value=False)

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        # 상단 탭 구성
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # --- Shepard Tone Tab ---
        self.shepard_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.shepard_frame, text="Shepard Tone")
        self.setup_shepard_tab()

        # --- Risset Rhythm Tab ---
        self.risset_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.risset_frame, text="Risset Rhythm")
        self.setup_risset_tab()

        # --- Plot Area ---
        self.plot_frame = ttk.LabelFrame(self.root, text="Signal Visualization")
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.fig, (self.ax_wave, self.ax_spec, self.ax_gram) = plt.subplots(3, 1, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- DAQ Control Area ---
        self.daq_frame = ttk.LabelFrame(self.root, text="DAQ Control")
        self.daq_frame.pack(fill="x", padx=10, pady=10)

        btn_finite = ttk.Button(self.daq_frame, text="Play Once", command=self.play_finite)
        btn_finite.pack(side="left", padx=10, pady=10)
        
        btn_loop = ttk.Button(self.daq_frame, text="Play Loop", command=self.play_continuous)
        btn_loop.pack(side="left", padx=10, pady=10)
        
        btn_stop = ttk.Button(self.daq_frame, text="STOP", command=self.stop_daq)
        btn_stop.pack(side="left", padx=10, pady=10)

        btn_quit = ttk.Button(self.daq_frame, text="QUIT", command=self.on_closing)
        btn_quit.pack(side="right", padx=10, pady=10)
        
        ttk.Label(self.daq_frame, text="Device:").pack(side="left", padx=(30, 5))
        self.device_entry = ttk.Entry(self.daq_frame, width=10)
        self.device_entry.insert(0, "Dev1")
        self.device_entry.pack(side="left", padx=5)

    def setup_shepard_tab(self):
        main_layout = ttk.Frame(self.shepard_frame)
        main_layout.pack(fill="both", expand=True, padx=20, pady=20)

        # Left: Controls
        ctrl_frame = ttk.Frame(main_layout)
        ctrl_frame.pack(side="left", fill="y", padx=20)

        self.s_duration = self.create_input(ctrl_frame, "Duration (s):", 5.0, 0)
        self.s_base_f = self.create_input(ctrl_frame, "Base Freq (Hz):", 50.0, 1)
        self.s_fs = self.create_input(ctrl_frame, "Sampling Rate (Hz):", self.fs_var.get(), 2)
        
        ttk.Label(ctrl_frame, text="Direction:").grid(row=3, column=0, sticky="w", pady=10)
        self.s_direction = tk.StringVar(value="1")
        ttk.Radiobutton(ctrl_frame, text="Ascending", variable=self.s_direction, value="1").grid(row=3, column=1, sticky="w")
        ttk.Radiobutton(ctrl_frame, text="Descending", variable=self.s_direction, value="-1").grid(row=4, column=1, sticky="w")

        ttk.Button(ctrl_frame, text="Generate Signal", command=self.generate_shepard).grid(row=5, column=0, columnspan=2, pady=30)

        # Right: Layer Selector & Options
        layer_frame = ttk.LabelFrame(main_layout, text="Active Layers")
        layer_frame.pack(side="left", fill="both", expand=True, padx=20)
        
        self.shepard_layer_container = ttk.Frame(layer_frame)
        self.shepard_layer_container.pack(fill="both", expand=True, padx=10, pady=10)
        self.refresh_layer_checkboxes("shepard", 5)

        options_frame = ttk.LabelFrame(main_layout, text="Feature Options")
        options_frame.pack(side="left", fill="both", expand=True, padx=20)
        
        ttk.Checkbutton(options_frame, text="Hann Envelope", variable=self.use_envelope).pack(anchor="w", padx=10, pady=10)
        ttk.Checkbutton(options_frame, text="Actuator Compensation", variable=self.use_compensation).pack(anchor="w", padx=10, pady=10)

    def setup_risset_tab(self):
        main_layout = ttk.Frame(self.risset_frame)
        main_layout.pack(fill="both", expand=True, padx=20, pady=20)

        # Left: Controls
        ctrl_frame = ttk.Frame(main_layout)
        ctrl_frame.pack(side="left", fill="y", padx=20)

        self.r_duration = self.create_input(ctrl_frame, "Duration (s):", 10.0, 0)
        self.r_tempo = self.create_input(ctrl_frame, "Base Tempo (BPM):", 60.0, 1)
        self.r_fs = self.create_input(ctrl_frame, "Sampling Rate (Hz):", self.fs_var.get(), 2)

        ttk.Label(ctrl_frame, text="Direction:").grid(row=3, column=0, sticky="w", pady=10)
        self.r_direction = tk.StringVar(value="1")
        ttk.Radiobutton(ctrl_frame, text="Accelerating", variable=self.r_direction, value="1").grid(row=3, column=1, sticky="w")
        ttk.Radiobutton(ctrl_frame, text="Decelerating", variable=self.r_direction, value="-1").grid(row=4, column=1, sticky="w")

        ttk.Button(ctrl_frame, text="Generate Signal", command=self.generate_risset).grid(row=5, column=0, columnspan=2, pady=30)

        # Right: Layer Selector & Options
        layer_frame = ttk.LabelFrame(main_layout, text="Active Layers")
        layer_frame.pack(side="left", fill="both", expand=True, padx=20)
        
        self.risset_layer_container = ttk.Frame(layer_frame)
        self.risset_layer_container.pack(fill="both", expand=True, padx=10, pady=10)
        self.refresh_layer_checkboxes("risset", 5)

        options_frame = ttk.LabelFrame(main_layout, text="Feature Options")
        options_frame.pack(side="left", fill="both", expand=True, padx=20)
        
        ttk.Checkbutton(options_frame, text="Hann Envelope", variable=self.use_envelope).pack(anchor="w", padx=10, pady=10)
        ttk.Checkbutton(options_frame, text="Actuator Compensation", variable=self.use_compensation).pack(anchor="w", padx=10, pady=10)

    def create_input(self, parent, label_text, default_val, row):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=10)
        entry = ttk.Entry(parent, width=10)
        entry.insert(0, str(default_val))
        entry.grid(row=row, column=1, sticky="ew", padx=5, pady=10)
        return entry

    def refresh_layer_checkboxes(self, mode, n=5):
        if mode == "shepard":
            container = self.shepard_layer_container
            vars_list = self.shepard_layer_vars
        else:
            container = self.risset_layer_container
            vars_list = self.risset_layer_vars

        # Clear existing
        for widget in container.winfo_children():
            widget.destroy()
        vars_list.clear()

        # Create new
        for i in range(n):
            var = tk.BooleanVar(value=True) # All active by default
            vars_list.append(var)
            cb = ttk.Checkbutton(container, text=f"Layer {i+1}", variable=var)
            cb.pack(anchor="w", pady=1)

    def update_plot(self, signal):
        fs = int(self.fs_var.get())
        # 1. Waveform (Time Domain)
        self.ax_wave.clear()
        duration = len(signal) / fs
        time_axis = np.linspace(0, duration, len(signal))
        step = max(1, len(signal) // 2000)
        self.ax_wave.plot(time_axis[::step], signal[::step], color="#2c3e50")
        self.ax_wave.set_title("Signal Waveform", fontsize=12)
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.grid(True, alpha=0.3)

        # 2. Power Spectrum (Frequency Domain) - Limit to 2000Hz
        self.ax_spec.clear()
        fft_data = np.abs(np.fft.rfft(signal))
        fft_freqs = np.fft.rfftfreq(len(signal), 1/fs)
        self.ax_spec.plot(fft_freqs, 20 * np.log10(fft_data + 1e-6), color="#e74c3c")
        self.ax_spec.set_title("Power Spectrum (FFT)", fontsize=12)
        self.ax_spec.set_ylabel("Magnitude (dB)")
        self.ax_spec.set_xlabel("Frequency (Hz)")
        self.ax_spec.set_xlim(0, 2000) # Show up to 2000Hz
        self.ax_spec.grid(True, alpha=0.3)

        # 3. Spectrogram (Time-Frequency Domain)
        self.ax_gram.clear()
        self.ax_gram.specgram(signal, Fs=fs, NFFT=1024, noverlap=512, cmap='viridis')
        self.ax_gram.set_title("Spectrogram", fontsize=12)
        self.ax_gram.set_ylabel("Frequency (Hz)")
        self.ax_gram.set_xlabel("Time (s)")
        
        self.fig.tight_layout()
        self.canvas.draw()

    def update_generators(self, fs):
        self.fs_var.set(str(fs))
        self.sig_gen.fs = fs
        self.risset_gen.fs = fs

    def generate_shepard(self):
        try:
            dur = float(self.s_duration.get())
            base_f = float(self.s_base_f.get())
            fs = int(self.s_fs.get())
            direction = int(self.s_direction.get())
            
            self.update_generators(fs)
            
            active = [i for i, var in enumerate(self.shepard_layer_vars) if var.get()]
            n_layers = len(self.shepard_layer_vars) # Max layers available
            
            self.current_signal = self.sig_gen.generate_shepard_tone(
                duration=dur, base_f=base_f, num_layers=n_layers, 
                direction=direction, active_layers=active,
                use_envelope=self.use_envelope.get()
            ) 
            self.update_plot(self.current_signal)
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")

    def generate_risset(self):
        try:
            dur = float(self.r_duration.get())
            tempo = float(self.r_tempo.get())
            fs = int(self.r_fs.get())
            direction = int(self.r_direction.get())
            
            self.update_generators(fs)
            
            active = [i for i, var in enumerate(self.risset_layer_vars) if var.get()]
            n_layers = len(self.risset_layer_vars)

            self.current_signal = self.risset_gen.generate_click_train(
                duration=dur, base_tempo=tempo, num_layers=n_layers,
                direction=direction, active_layers=active,
                use_envelope=self.use_envelope.get()
            )
            self.update_plot(self.current_signal)
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")

    def init_daq(self):
        if self.daq is None:
            dev = self.device_entry.get()
            try:
                fs = int(self.fs_var.get())
                self.daq = DAQOutput(device_name=dev, sampling_rate=fs)
            except Exception as e:
                messagebox.showerror("DAQ Error", f"Check hardware/Device Name: {e}")
                return False
        return True

    def play_finite(self):
        if self.current_signal is None: return
        if self.init_daq():
            self.daq.send_signal_finite(self.current_signal)

    def play_continuous(self):
        if self.current_signal is None: return
        if self.init_daq():
            self.daq.send_signal_continuous(self.current_signal)

    def stop_daq(self):
        if self.daq:
            self.daq = None
            messagebox.showinfo("Info", "DAQ Task Stopped.")

    def on_closing(self):
        self.stop_daq()
        self.root.destroy()
        sys.exit()

if __name__ == "__main__":
    root = tk.Tk()
    app = ShepardApp(root)
    root.mainloop()
