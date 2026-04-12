import numpy as np

class SigGen():

    def __init__(self):
        self.HpS = 100 # Hertz increament per Second
        self.BaseFreq = 50 # Base Freq
        self.Layer = 5
        self.Metabar = 1
        self.TotalDuration = 10
        self.SamplingRate = 10000
        self.Final_Signal = np.zeros((self.Layer, self.TotalDuration * self.SamplingRate))

    def PerceivedIntensity(self):
        PI = np.zeros((self.Layer, self.TotalDuration * self.SamplingRate))

        for i in range(self.Layer):
            PI[i] = 1



    def Shepard(self):


    def generate(self, cond):
        # self.Final_Signal

        return self.Final_Signal