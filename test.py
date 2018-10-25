import keras

import numpy as np
import cv2
import os

from train_lstm import build_lstm_net
from load_data import associate_camptr__to__speed, get_flist

#---
#--- Build model and load trained weights
#---
model = build_lstm_net(input_shape=(60, 160, 320, 3) )
print 'Open pretrained model `model_speed.keras`'
model.load_weights( 'model_speed.keras')


#---
#--- Load a text sequence
#---
cam_files, log_files = get_flist( COMMA_PATH='/Bulk_Data/cv_datasets/comma/comma-dataset/')
X, speeds = associate_camptr__to__speed( cam_files[6], log_files[6] )

assert( X.shape[0] == len(speeds) )
for i in range( 60, X.shape[0] ):
    print 'i=%d' %(i)

    # Get Previous 60 images for prediction
    u = np.expand_dims( X[i-60:i,:,:,:], 0 ) #1x60x160x320x3
    pred_out = model.predict( u )
    print "pred_out", pred_out.argmax()
    predicted_speed = np.dot(np.arange( 0, 35, 0.5 ) , pred_out[0])


    # Display Image and Overlay predicted Value
    im = np.array( X[i,:,:,:].astype('uint8') )

    imC = np.ascontiguousarray( im[:,:,::-1] )

    # Put ground-truth
    cv2.putText(imC,"#%d" %(i), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    cv2.putText(imC,"ground-truth  : %4.2f" %(speeds[i]), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
    cv2.putText(imC,"predicted-lstm: %4.2f" %( predicted_speed ), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    cv2.imwrite( '/Bulk_Data/cv_datasets/comma/result/%06d.jpg' %(i), imC )

    cv2.imshow( 'win', imC )
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
