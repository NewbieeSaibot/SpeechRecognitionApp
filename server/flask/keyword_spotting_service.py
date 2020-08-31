import tensorflow.keras as keras
import librosa
import numpy as np

NUM_SAMPLES_TO_CONSIDER = 22050
MODEL_PATH = "./model.h5"


class _Keyword_Spotting_Service():

    model = None
    _mappings = ["bed", "bird", "cat", "dog", "down", "eight", "five", "four", "go",
                 "happy", "house", "left", "marvin", "nine", "no", "off", "on", "one", "right", "seven",
                 "sheila", "six", "stop", "three", "tree", "two", "up", "wow", "yes", "zero", "background_noise"]
    _istance = None

    def predict(self, file_path):
        #extract the mfccs
        MFCCs = self.preprocess(file_path) #(segments, 13)

        #convert 2d mfccs array into 4d array -> (number of samples,segments, 13, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediction
        predictions = self.model.predict(MFCCs) #[[0.1, 0.6, 0.1, ....]]
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length = 512):

        # load audio file
        signal, sr = librosa.load(file_path)

        #ensure consistency audio file length
        if(len(signal) > NUM_SAMPLES_TO_CONSIDER):
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length= hop_length)

        return MFCCs.T


def Keyword_Spotting_Service():

    #ensure that we only have 1 instance of kss
    if _Keyword_Spotting_Service._istance is None:
        _Keyword_Spotting_Service._istance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)

    return _Keyword_Spotting_Service._istance

if __name__ == "__main__":

    kss = Keyword_Spotting_Service()
    print(kss.predict("./dataset/cat/0ab3b47d_nohash_0.wav"))
