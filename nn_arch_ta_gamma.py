import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import h5py as h5

# dropout rate
dr_rate = 0.1
# optional L2 regularization
regul = 0
reg_factor_last = 0
# weights for proton- and photon-induced events, respectively
weights = np.array([1.,5.]).astype(np.float32)

## NN layers

# layers for initializing lstm states
def make_lstm_init(out):

    lstm_init = keras.Input(shape=(10))

    x = layers.Dense(16,
                    kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul))(lstm_init)
    x = tf.keras.activations.gelu(x)
    x = layers.Dense(out,
                    kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul))(x)
    state1 = tf.keras.activations.gelu(x)
    y = layers.Dense(16,
                    kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul))(lstm_init)
    y = tf.keras.activations.gelu(y)
    y = layers.Dense(out,
                    kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul))(y)
    state2 = tf.keras.activations.gelu(y)

    model = keras.Model(lstm_init, (state1,state2), name='lstm_state_creator'+str(out))
    return model

def make_lstm_full(out):

    lstm_init = keras.Input(shape=(5))

    x = layers.Dense(16,
                    kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul))(lstm_init)
    x = tf.keras.activations.gelu(x)
    x = layers.Dense(out,
                    kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul))(x)
    state1 = tf.keras.activations.gelu(x)
    y = layers.Dense(16,
                    kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul))(lstm_init)
    y = tf.keras.activations.gelu(y)
    y = layers.Dense(out,
                    kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul))(y)
    state2 = tf.keras.activations.gelu(y)

    model = keras.Model(lstm_init, (state1,state2), name='lstm_state_creator'+str(out))
    return model

# lstm used in ecoder for analysing waveforms
def add_rnn(wf_shape):

    wf = keras.Input(shape=wf_shape)
    lstm_init = keras.Input(shape=(10))

    state_cr_1 = make_lstm_init(20)
    lstm_state_1 = state_cr_1(lstm_init)
    state_cr_2 = make_lstm_init(16)
    lstm_state_2 = state_cr_2(lstm_init)

    lstm_layer_1 = tf.keras.layers.LSTM(20, activation='tanh', recurrent_activation='sigmoid', return_sequences=True,
                                kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul))
    lstm_layer_2 = tf.keras.layers.LSTM(16, activation='tanh', recurrent_activation='sigmoid',
                                kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul))
    
    x = lstm_layer_1(wf, initial_state=lstm_state_1)
    x = layers.BatchNormalization()(x)
    x = lstm_layer_2(x, initial_state=lstm_state_2)
    wf_enc = layers.BatchNormalization()(x)

    model = keras.Model((wf,lstm_init), wf_enc, name='wf_rnn_encoder')
    return model

# waveform encoder
def make_wf_encoder(dr_rate):
    
    wf = keras.Input(shape=(128,2))
    lstm_init = keras.Input(shape=(10))

    x = tf.expand_dims( wf, axis=-1 )

    x = tf.keras.layers.Conv2D( filters=14, kernel_size=(6,1), strides=(1,1), padding='same',
                                kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul))(x)
    x = tf.keras.layers.Dropout(dr_rate)(x)
    x = tf.keras.activations.gelu(x)
    x = layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D( filters=16, kernel_size=(6,2), strides=(1,1), padding='valid',
                                kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul))(x)
    x = tf.keras.layers.Dropout(dr_rate)(x)
    x = tf.keras.activations.gelu(x)
    x = layers.BatchNormalization()(x)

    x = tf.squeeze( x, axis=-2 )

    x = tf.keras.layers.Conv1D( filters=20, kernel_size=6, strides=2, padding='valid',
                                kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul))(x)
    x = tf.keras.layers.Dropout(dr_rate)(x)
    x = tf.keras.activations.gelu(x)
    x = layers.BatchNormalization()(x)

    te = tf.expand_dims(x, axis=2)
    shape =te.get_shape().as_list()[1:4:2]
    rnn = add_rnn(shape)
    
    x = rnn((x, lstm_init))

    wf_enc = layers.Dense(6,
                         kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul))(x)

    model = keras.Model((wf,lstm_init), wf_enc, name='wf_encoder')
    return model

# spatial bundle
def make_dt_encoder(dr_rate):
    
    encoder_input = keras.Input(shape=((6,6,7)))

    x = tf.keras.layers.Conv2D( filters=16, kernel_size=(3,3), strides=(1,1), padding='same',
                                kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul) )(encoder_input)
    x = tf.keras.layers.Dropout(dr_rate)(x)
    x = tf.keras.activations.gelu(x)
    x = layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D( filters=20, kernel_size=(3,3), strides=(1,1), padding='same',
                                kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul) )(x)   
    x = tf.keras.layers.Dropout(dr_rate)(x)
    x = tf.keras.activations.gelu(x)  
     
    x = tf.keras.layers.AveragePooling2D( pool_size=(2,2), padding='valid')(x)
    x = layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Conv2D( filters=20, kernel_size=(2,2), strides=(1,1), padding='same',
                                kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul) )(x)
    x = tf.keras.layers.Dropout(dr_rate)(x)
    x = tf.keras.activations.gelu(x) 
    x = layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D( filters=20, kernel_size=(2,2), strides=(1,1), padding='valid',
                                kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul) )(x)
    x = tf.keras.layers.Dropout(dr_rate)(x)
    x = tf.keras.activations.gelu(x) 
    x = layers.BatchNormalization()(x)     

    x = layers.Flatten()(x)
    dt_enc = layers.Dense(6,
                          kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul))(x)
  
    model = keras.Model(encoder_input, dt_enc, name='detectors_bundle_encoder')
    return model

# temorial bundle
def make_wfs_analyzer(dr_rate):
    
    wf_fl = keras.Input(shape=(num_flat,128,2))
    mask_fl = keras.Input(shape=(num_flat,), dtype=bool)
    dt_fl = keras.Input(shape=(num_flat,6))
    lstm_init = keras.Input(shape=(num_flat,10))

    wf_encoder = make_wf_encoder(dr_rate)
    lstm_layer = tf.keras.layers.LSTM(20, activation='tanh', recurrent_activation='sigmoid',
                                kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul))
    bidir = tf.keras.layers.Bidirectional( lstm_layer, merge_mode='mul' )
    
    wf_resh = tf.reshape(wf_fl, (-1,128,2))
    mask_resh = tf.reshape(mask_fl, (-1,1))
    lstm_init_resh = tf.reshape(lstm_init, (-1,10))
    wf_encs = tf.where( mask_resh, wf_encoder((wf_resh,lstm_init_resh)), tf.zeros(6) )
    #wf_encs = wf_encoder((wf_resh,lstm_init_resh))*tf.cast(mask_fl, tf.float32)
    wf_encs = tf.reshape(wf_encs, (-1,tf.shape(dt_fl)[1],6))
    
    wf_dt = tf.concat((wf_encs,dt_fl), axis=-1)
    #wf_dt = tf.squeeze(wf_dt)

    lstm_st_creator = make_lstm_full(20)
    ltsm_init_full = lstm_init[:,0,:5]
    lstm_st_full = lstm_st_creator(ltsm_init_full)

    rnn_res = bidir(inputs=wf_dt, initial_state=[lstm_st_full[0],lstm_st_full[1],lstm_st_full[0],lstm_st_full[1]], mask=mask_fl) # if slow, *mask_fl the result
    
    wfs_encoded = layers.Dense(8,
                              kernel_regularizer=tf.keras.regularizers.L2(regul), bias_regularizer=tf.keras.regularizers.L2(regul))(rnn_res)

    model = keras.Model((wf_fl,mask_fl,dt_fl,lstm_init), wfs_encoded, name='wfs_encoder')
    return model

# combined analysis
def make_union_analyzer(input_shape):
    
    d_in = keras.Input(shape=input_shape)
    x = d_in

    units = [32, 24, 16]
    drops = [0.1, 0.0, 0.0]
    
    for (unit,dr) in zip(units,drops): 
        x = layers.Dense(unit, kernel_regularizer=tf.keras.regularizers.L2(reg_factor_last*regul), bias_regularizer=tf.keras.regularizers.L2(reg_factor_last*regul))(x)
        x = tf.keras.layers.Dropout(dr)(x)
        x = tf.keras.activations.gelu(x)
        x = layers.BatchNormalization()(x)
        
    preds = layers.Dense(2, activation="softmax", kernel_regularizer=tf.keras.regularizers.L2(reg_factor_last*regul), bias_regularizer=tf.keras.regularizers.L2(reg_factor_last*regul))(x)
    
    model = keras.Model(d_in, preds, name='final_analyzer')    
    return model

# combining layers
def make_classifier(dr_rate, sm_ker_size, sm_factor):
    
    dt_fl_in = keras.Input(shape=(num_flat,7))
    wfs_in = keras.Input(shape=(num_flat,128,2))
    gp_in = keras.Input(shape=(15))
    dt_in = keras.Input(shape=(6,6,7))
    
    mask_wfs_in = tf.cast( dt_fl_in[:,:,-1], bool )
    
    dt_an = make_dt_encoder(dr_rate)
    dt_enc = dt_an(dt_in)
    
    wfs_an = make_wfs_analyzer(dr_rate)
        
    gp_to_lstm = tf.expand_dims( tf.concat( (gp_in[:,0:3],gp_in[:,6:7],gp_in[:,9:10]), axis=-1), axis=1)
    gp_to_lstm = tf.repeat(gp_to_lstm,repeats=tf.shape(dt_fl_in)[1],axis=1)
    lstm_init = tf.concat( (dt_fl_in[:,:,0:3],dt_fl_in[:,:,4:6], gp_to_lstm), axis=-1 )
    wfs_enc = wfs_an((wfs_in,mask_wfs_in,dt_fl_in[:,:,:-1],lstm_init))
    
    un_enc = tf.concat((gp_in,dt_enc,wfs_enc), axis=-1)
    un_an = make_union_analyzer(un_enc.shape[1:])
    pred = un_an(un_enc)
    
    model = keras.Model( inputs=(dt_fl_in, wfs_in, gp_in, dt_in), outputs=pred, name='classifier')   
    return model