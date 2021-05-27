# Digit Audio Classification 🔢

`A Deep Learning model that classifies digits based on its Audio Input.`

## Audio 🔊
Audio is a term used to describe any sound or noise in a range the human ear is capable of hearing. Measured in `hertz`, the audio signal on a computer is generated using a sound card and heard through speakers or headphones.

Any digital information with speech or music stored on and played through a computer is known as an audio file or sound file. One of the most common types of audio file formats used today is the `MP3`.

An audio signal is a representation of sound, typically using either a changing level of electrical voltage for analog signals, or a series of binary numbers for digital signals.


## Waveplots 〰
To plot all these waveplots, I came up across an exciting python library `librosa`. `librosa` is a python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems.

These are the waveplots of the digits when we spell them. Please note that we are spelling `one` or any other digit as it should be spelled, not like `oooooaannee` or something like that. You can also try it once. Its funny though. 😆😆😆😆

### Zero 0️⃣

<p align=center>
  <img src="https://github.com/Ankit152/DigitAudio/blob/main/img/0.png" width=1000>
</p>


### One 1️⃣

<p align=center>
  <img src="https://github.com/Ankit152/DigitAudio/blob/main/img/1.png" width=1000>
</p>


### Two 2️⃣

<p align=center>
  <img src="https://github.com/Ankit152/DigitAudio/blob/main/img/2.png" width=1000>
</p>

### Three 3️⃣

<p align=center>
  <img src="https://github.com/Ankit152/DigitAudio/blob/main/img/3.png" width=1000>
</p>

### Four 4️⃣

<p align=center>
  <img src="https://github.com/Ankit152/DigitAudio/blob/main/img/4.png" width=1000>
</p>

### Five 5️⃣

<p align=center>
  <img src="https://github.com/Ankit152/DigitAudio/blob/main/img/5.png" width=1000>
</p>

### Six 6️⃣

<p align=center>
  <img src="https://github.com/Ankit152/DigitAudio/blob/main/img/6.png" width=1000>
</p>

### Seven 7️⃣

<p align=center>
  <img src="https://github.com/Ankit152/DigitAudio/blob/main/img/7.png" width=1000>
</p>

### Eight 8️⃣

<p align=center>
  <img src="https://github.com/Ankit152/DigitAudio/blob/main/img/8.png" width=1000>
</p>

### Nine 9️⃣

<p align=center>
  <img src="https://github.com/Ankit152/DigitAudio/blob/main/img/9.png" width=1000>
</p>


## Countplot 📊

**This is the countplot of the audio data. The X-axis represents the digits 0-9 and the Y-axis represents the count of respective digit.**
<p align=center>
  <img src="https://github.com/Ankit152/DigitAudio/blob/main/img/countplotAudio.png" width=1000>
</p>

*We can note that the dataset is equally distributed i.e. they are balanced.*

## Model Architecture 🌟

* `Layer 0`: Input of 40 dimensional
* `Layer 1`: Dense layer of 128 neurons followed by ReLU Activation and Dropout rate of 0.25.
* `Layer 2`: Dense layer of 256 neurons followed by ReLU Activation and Dropout rate of 0.25.
* `Layer 3`: Dense layer of 128 neurons followed by ReLU Activation and Dropout rate of 0.25.
* `Layer 4`: The softmax layer for the classifiaction of 10 different digits.

**The below is the pictoral representation of the model architecture.**

<p align=center>
  <img src="https://github.com/Ankit152/DigitAudio/blob/main/img/graph.png" width=1000>
</p>

## Model Training ⚔️

The model was trained over 200 epochs. Here are the `accuracy` and `loss` curves.

### Accuracy vs Epochs 📈

<p align=center>
  <img src="https://github.com/Ankit152/DigitAudio/blob/main/img/accuracy.png" width=1000>
</p>

### Loss vs Epochs 📉

<p align=center>
  <img src="https://github.com/Ankit152/DigitAudio/blob/main/img/loss.png" width=1000>
</p>

*From the above figure we can say that the model is not overfitting.*
