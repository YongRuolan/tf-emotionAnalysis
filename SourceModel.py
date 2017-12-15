from random import shuffle

from keras import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Masking, Bidirectional, LSTM, TimeDistributed, Dense

num_class = 7

class bilstmAttention():
    def __init__(self, sequenceLength=None):
        self.seqenceLength = sequenceLength
        return

    def model(self):
        model = Sequential()
        model.add(Masking(mask_value= -1,input_shape=(self.sequenceLength, 23*3,)))
        model.add(Bidirectional(LSTM(100, dropout_W=0.2, dropout_U=0.2, input_shape=(self.sequenceLength, 23 * 3,))))
        model.add(TimeDistributed(Dense(num_class, activation='softmax',name="softmax")))
        model.summary()
        return model

    def train(self,train_gen,valid_gen,samples_per_epoch,validation_steps,epoch):
        model = self.model()
        model.compile(loss='categorical_crossentropy', optimizer='sgd')

        # define the checkpoint
        filepath = "newresults/weightImprovements-{epoch:02d}-{loss:.4f}.hdf5"
        # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        checkpoint = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto',
                                       epsilon=0.0001, cooldown=0, min_lr=0)
        callbacks_list = [checkpoint]
        model.fit_generator(train_gen,
                            samples_per_epoch = samples_per_epoch,
                            validation_data=valid_gen,
                            validation_steps=validation_steps,
                            nb_epoch=epoch,
                            verbose=1,
                            callbacks=callbacks_list)

    def  read_by_batch(df,n_training):
        while(1):
            df = shuffle(df)
            for i in range(0,n_training,batch_size):
                X=df[i:i+batch_size].as_matrix()[:,0:seq_length]
                Y = df[i:i + batch_size].as_matrix()[:,1:]
                Y = Y.tolist()
                for i in range(len(Y)):
                    Y[i] = np_utils.to_categorical(Y[i], num_classes=n_vocab + 1)
                Y = np.asarray(Y)
                yield X,Y



