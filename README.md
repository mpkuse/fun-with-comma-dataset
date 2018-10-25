# Fun with Comma Dataset.

A quick and simple experiement with LSTMs. Learning to predict speed of the vehicles
from dash-cam video. I used the comma.ai dataset for training an LSTM.

## How to obtain the dataset?
You may obtain the full dataset here. [Here](https://github.com/commaai/research)
However for the purpose of this quick experiement, I have uploaded 1 video sequence
[https://mega.nz/#F!14FEBYzA!mToOvUT-9Yq0hU2l15PZjA](https://mega.nz/#F!14FEBYzA!mToOvUT-9Yq0hU2l15PZjA). Click on this link and then click 'download as ZIP'. The link will expire by 1st Nov. To be used only for
non-commercial purpose.

## Requirement
- Tensorflow 1.8+
- Keras 2.0+
- OpenCV 2.4+
- numpy

## Video
[Video]( https://youtu.be/LzXRh2Pe740 )


## Training
Besure set the datapath COMMA_PATH in train_lstm.py.
```
python train_lstm.py
```

## Testing
Besure set the datapath COMMA_PATH in test.py
```
python test.py
```

## Details
[Model][images/model.png]
