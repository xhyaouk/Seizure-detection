
import argparse,sklearn,math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,scale
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import os

import gc
from random import shuffle


from keras.layers import merge, Multiply, TimeDistributed, GlobalAveragePooling1D
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras import backend
from keras import optimizers
from keras import regularizers
from keras import initializers
from keras.layers import Bidirectional



# ************************************************************
# options
# ************************************************************
parser = argparse.ArgumentParser()
parser.add_argument('--L', type=int, default=17)
parser.add_argument('--validate-test-set-size', type=float, default=0.3)
parser.add_argument('--learning-rate', type=float, default=0.0013)
parser.add_argument('--lstm-units', type=int, default=140)
parser.add_argument('--dense-units', type=int, default=70)
parser.add_argument('--epochs', type=int, default=35)
parser.add_argument('--batch-size', type=int, default=30)
parser.add_argument('--normalize', action='store_true', default=False)

parser.add_argument('--single_attention_vector', action='store_true', default=True)
parser.add_argument('--mode',default='concat')  #Bidirectional mode: 'concat', 'sum', 'ave','mul'

if __name__ == '__main__' and '__file__' in globals():
    options = parser.parse_args()
else:
    options = parser.parse_args([])


# ************************************************************
# load data
# ************************************************************

def get_array_data():   # to obtain train_data, train_label, test_data, test_label, validate_data, validate_label
    path = '/home/CHB_MIT_data/store_data_5/'

    files = os.listdir(path)
    seizure_file_name_list = list(filter(lambda x:x.endswith('_seizure.npy'),files))
    seizure_file_name_list.sort()

    non_seizure_file_name_list = list(filter(lambda x:x.endswith('_nonseizure.npy'),files))
    non_seizure_file_name_list.sort()

    seizure_data_list = []
    non_seizure_data_list = []
    label_list = []

    for file in seizure_file_name_list:
        A = np.load(path+file)[:,0:-1]
        seizure_data_list = seizure_data_list + list(A)

    num_seizure_segments = len(seizure_data_list)


    for file in non_seizure_file_name_list:
        A = np.load(path+file)[:,0:-1]
        non_seizure_data_list = non_seizure_data_list + list(A)

    index_selection = list(np.random.permutation(len(non_seizure_data_list))[0:num_seizure_segments])
    non_seizure_selection = []
    
    for j in index_selection:
        non_seizure_selection.append(non_seizure_data_list[j])

    non_seizure_data_list = non_seizure_selection

    assert len(seizure_data_list) == len(non_seizure_data_list)

    for i in range(num_seizure_segments):
        assert set(seizure_data_list[i][-1]) == {'seizure'}
        assert set(non_seizure_data_list[i][-1]) == {'non_seizure'}   

    seizure_train_data_list, seizure_validate_test_data_list, non_seizure_train_data_list, non_seizure_validate_test_data_list = train_test_split(
        seizure_data_list,non_seizure_data_list,test_size=options.validate_test_set_size)

    seizure_validate_data_list, seizure_test_data_list, non_seizure_validate_data_list, non_seizure_test_data_list = train_test_split(
        seizure_validate_test_data_list,non_seizure_validate_test_data_list,test_size=0.5)

    
    # According to the ratio 7:1.5:1.5, split seizure_data_list to get seizure_train_data_list, seizure_test_data_list and seizure_validate_data_list.
    # And so does non_seizure_data_list.
    # In such a way, we hope to get equal seizure samples and non-seizure samples in the training data set, obtain equal seizure samples and non-seizure
    # samples in the testing data set, and obtain equal seizure samples and non-seizure samples in the validation data set.

    print('number of seizure_train_data:',len(seizure_train_data_list))
    print('number of non_seizure_train_data:',len(non_seizure_train_data_list))
    print('number of seizure_validate_data:',len(seizure_validate_data_list))
    print('number of non_seizure_validate_data:',len(non_seizure_validate_data_list))
    print('number of seizure_test_data:',len(seizure_test_data_list))
    print('number of non_seizure_test_data:',len(non_seizure_test_data_list))

    del(seizure_data_list)
    del(non_seizure_data_list)
    gc.collect()
    
    train_data_list = seizure_train_data_list + non_seizure_train_data_list
    test_data_list = seizure_test_data_list + non_seizure_test_data_list
    validate_data_list  = seizure_validate_data_list + non_seizure_validate_data_list

    shuffle(train_data_list)
    shuffle(test_data_list)
    shuffle(validate_data_list)

    train_label_list = []
    test_label_list = []
    validate_label_list = []

    for i in range(len(train_data_list)):
        if set(train_data_list[i][-1]) == {'seizure'} :
            train_label_list.append(1)
        else:
            train_label_list.append(0)

    for i in range(len(test_data_list)):
        if set(test_data_list[i][-1]) == {'seizure'}:
            test_label_list.append(1)
        else:
            test_label_list.append(0)

    for i in range(len(validate_data_list)):
        if set(validate_data_list[i][-1]) == {'seizure'} :
            validate_label_list.append(1)
        else:
            validate_label_list.append(0)

    train_data = np.array(train_data_list)[:,0:-1]
    test_data = np.array(test_data_list)[:,0:-1]
    validate_data = np.array(validate_data_list)[:,0:-1]

    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    validate_data = validate_data.astype('float32')

    train_label = np.array(train_label_list).astype('int32')
    test_label = np.array(test_label_list).astype('int32')
    validate_label = np.array(validate_label_list).astype('int32')

    assert len(train_data) == len(train_label)
    assert len(test_data) == len(test_label)
    assert len(validate_data) == len(validate_label)

    return train_data, train_label, test_data, test_label, validate_data, validate_label


X_train,Y_train, X_test, Y_test, X_validate, Y_validate = get_array_data()

X_train_list = list(X_train)
X_test_list = list(X_test)
X_validate_list = list(X_validate)

if options.normalize:
    for i in range(len(X_train_list)):
        X_train_list[i] = scale(X_train_list[i], axis=0)
        
    for j in range(len(X_test_list)):
        X_test_list[j] = scale(X_test_list[j], axis=0)

    for j in range(len(X_validate_list)):
        X_validate_list[j] = scale(X_validate_list[j], axis=0)
        

X_train = np.array(X_train_list,dtype='float32')
X_test = np.array(X_test_list,dtype='float32')
X_validate = np.array(X_validate_list, dtype='float32')

print('X_train:',type(X_train),X_train.shape,X_train.dtype)
print('Y_train:',type(Y_train),Y_train.shape,Y_train.dtype)
print('X_test:',X_test.shape)
print('Y_test:',Y_test.shape)
print('X_validate:',X_validate.shape)
print('Y_validate:',Y_validate.shape)

assert X_train.shape[2] == options.L

nC = 2

print('number of 1 in Y_test:',sum(Y_test))



# ************************************************************
# construct model
# ************************************************************

timesteps = X_train.shape[1]

cons_initializer_1 = initializers.Constant(value=0.1)

truncated_normal_initializer_2 = initializers.TruncatedNormal(mean=0.0,stddev=0.1)

glorot_uniform_initializer_3 = initializers.glorot_uniform()

def attention_module(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim_1 = int(inputs.shape[1])
    input_dim_2 = int(inputs.shape[2])
    
    a = Dense(input_dim_2, activation='softmax',
              kernel_initializer=truncated_normal_initializer_2,
              kernel_regularizer=regularizers.l2(l=0.02),
              bias_regularizer=regularizers.l2(l=0.02)
              )(inputs)
    
    if options.single_attention_vector:
        a = Lambda(lambda x: backend.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim_1)(a)
    
    output_attention_mul = Multiply()([inputs, a])
    return output_attention_mul


inputs = Input(shape=(timesteps,options.L))
attention_mul = attention_module(inputs)

bilstm_out = Bidirectional(LSTM(options.lstm_units,return_sequences=True),merge_mode=options.mode)(attention_mul)

#lstm_out = LSTM(options.lstm_units,return_sequences=True)(attention_mul)

distributed_compute = TimeDistributed(Dense(options.dense_units,
                                            kernel_regularizer=regularizers.l2(l=0.02),
                                            bias_regularizer=regularizers.l2(l=0.02)
                                            ))(bilstm_out)

pooling_out = GlobalAveragePooling1D()(distributed_compute)

outputs = Dense(nC,activation='softmax',
                kernel_initializer=glorot_uniform_initializer_3,
                bias_initializer=cons_initializer_1,
                kernel_regularizer=regularizers.l2(l=0.02),
                bias_regularizer=regularizers.l2(l=0.02)
                )(pooling_out)

model = Model(input=inputs,output=outputs)

RMSprop = optimizers.RMSprop(lr=options.learning_rate)
Nadam = optimizers.Nadam(lr=options.learning_rate)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=RMSprop,
              metrics=['accuracy'])

print(model.summary())

model.fit(X_train, Y_train,
          validation_data=(X_validate, Y_validate),
          batch_size=options.batch_size,
          epochs=options.epochs)

y_pred = model.predict(X_test)
y_pred_label = np.argmax(y_pred,axis=1)

test_accuracy = np.sum(y_pred_label == Y_test)/Y_test.shape[0]
print('original test accuracy:',test_accuracy)

cm = confusion_matrix(Y_test,y_pred_label)
print('confusion matrix:',cm)


def compute_cm_2(confu_mat,num_classes=2):  # metric-computing way corresponding to the case of 2 classes
    TP = confu_mat[1,1]
    FP = confu_mat[0,1]
    FN = confu_mat[1,0]
    TN = confu_mat[0,0]

    sensitivity = TP / (TP+FN)
    specificity = TN / (TN+FP)
    F1 = 2*TP / (2*TP+FP+FN)
    precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + FN + TN + FP)
    return sensitivity, specificity, F1, precision, accuracy

sensitivity, specificity, F1, precision, accuracy = compute_cm_2(cm)

string = 'Test results using comfusion matrix: Sensitivity: {:.4f}, Specificity: {:.4f}, F1: {:.4f}, Precision: {:.4f}, Accuracy: {:.4f}'.format(
    sensitivity, specificity, F1, precision, accuracy)

print(string)








    




