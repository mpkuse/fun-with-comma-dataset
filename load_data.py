import h5py
import numpy as np
import cv2
import glob
import code

import keras

def associate_camptr__to__speed( cam_file, log_file ):

    print 'Open File: ', cam_file
    h5_cam = h5py.File( cam_file )

    print 'Open File: ', log_file
    h5_log = h5py.File( log_file )

    # >>> h5_cam.keys()
    # [u'X']
    X = h5_cam['X'].value
    X = np.transpose( X, (0,2,3,1) ) #I like channels_last format
    print 'X.shape', X.shape

    # >>> h5_log.keys()
    # [u'UN_D_cam1_ptr', u'UN_D_cam2_ptr', u'UN_D_camphone_ptr', u'UN_D_lidar_ptr', u'UN_D_radar_msg',
    #   u'UN_D_rawgps', u'UN_T_cam1_ptr', u'UN_T_cam2_ptr', u'UN_T_camphone_ptr', u'UN_T_lidar_ptr',
    #   u'UN_T_radar_msg', u'UN_T_rawgps', u'blinker', u'brake', u'brake_computer', u'brake_user',
    #   u'cam1_ptr', u'cam2_ptr', u'camphone_ptr', u'car_accel', u'fiber_accel', u'fiber_compass',
    #   u'fiber_compass_x', u'fiber_compass_y', u'fiber_compass_z', u'fiber_gyro', u'fiber_temperature',
    #   u'gas', u'gear_choice', u'idx', u'rpm', u'rpm_post_torque', u'selfdrive', u'speed',
    #   u'speed_abs', u'speed_fl', u'speed_fr', u'speed_rl', u'speed_rr', u'standstill',
    #   u'steering_angle', u'steering_torque', u'times', u'velodyne_gps', u'velodyne_heading', u'velodyne_imu']


    #---
    #--- Extract Speeds from the log as per cam1_ptr and speed.
    #---
    G = zip( h5_log['cam1_ptr'].value.astype('int32'), h5_log['speed'].value )
    G2 = np.zeros( (X.shape[0],2) ) #1st column is sum of speeds, 2nd col is number of instances

    for i, g in enumerate( G ):
        if i%10000 == 0:
            print 'processing log#%d of %d' %(i, len(G) )

        # g0: cam_ptr, g1: speed
        G2[g[0],0] += g[1]
        G2[g[0],1] += 1


    G2[:,0] = G2[:,0] / G2[:,1]
    # G2 has avg speeds at every frame. Save this file.

    # Return X, G2[:,0]
    return X[2000:-2000], G2[:,0][2000:-2000]


def get_flist(COMMA_PATH):
    # COMMA_PATH = '/Bulk_Data/cv_datasets/comma/comma-dataset/'


    cam_files = glob.glob( COMMA_PATH+'/camera/*.h5' )
    log_files = []
    for f in cam_files:
        spl = f.split('/')
        assert( spl[-2] == 'camera' )
        spl[-2] = 'log'
        f_log = '/'.join( spl )
        log_files.append( f_log )
    print 'n_files = ', len(cam_files )
    return cam_files, log_files

class KerasBatchGenerator( keras.utils.Sequence ):
    def __init__(self, COMMA_PATH, training=True):
        cam_files, log_files = get_flist(COMMA_PATH)

        # For training use 1st 8 files
        if training:
            cam_files = cam_files[0:8]
            log_files = log_files[0:8]
        else:
            cam_files = cam_files[8:]
            log_files = log_files[8:]

        self.batch_size = 16
        self.t_step = 60
        self.epochs = 0

        self.X, self.speeds = associate_camptr__to__speed( cam_files[self.epochs], log_files[self.epochs] )


    def __len__(self):
        return self.X.shape[0] // ( self.batch_size + self.t_step )

    def __getitem__( self, idx ):
        IM = []
        Y = []

        e_start = idx*(self.batch_size+self.t_step)

        for i in range( self.batch_size ):
            s = e_start+i
            e = e_start+i+self.t_step
            # print 's=', s, '\te=', e

            IM.append( self.X[s:e] )
            Y.append( self.speeds[e+1] )

        IM = np.array( IM )
        Y = np.array( Y )

        # Make 1-hot representation for Y. 70-class problem
        Y_classes = np.ceil( 2.*np.clip( Y, 0, 35 )).astype('int32')

        return IM, keras.utils.to_categorical(Y, 70)

    def on_epoch_end(self):
        # Every epoch change file
        self.epochs += 1

        print 'epochs=', self.epochs, '\t so load next dataset'
        self.X, self.speeds = associate_camptr__to__speed( cam_files[self.epochs%len(cam_files)], log_files[self.epochs%len(cam_files)] )







if __name__ == '__main__':
    gen = KerasBatchGenerator(COMMA_PATH='/Bulk_Data/cv_datasets/comma/comma-dataset/')

if __name__ == '__1main__':
    COMMA_PATH='/Bulk_Data/cv_datasets/comma/comma-dataset/'
    cam_files, log_files = get_flist(COMMA_PATH)

    for indx in range( len(cam_files) ):
        cam_file = cam_files[indx]
        log_file = log_files[indx]

        X, speeds = associate_camptr__to__speed( cam_file, log_file )

        for i in range( 0,  X.shape[0], 100 ):
            print 'Showing frame#%d of %d\t speed=%4.4fm/s' %(i, X.shape[0], speeds[i] )
            cv2.imshow( 'win', X[i,:,:,::-1] )
            key = cv2.waitKey(10)
            if key == ord( 'q' ):
                break
