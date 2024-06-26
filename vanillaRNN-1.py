# just simple code file to show the default structure of embedding & simpleRNN layers model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Sequential

def build_model1():
    model = Sequential()
    
    model.add(layers.Embedding(10, 5))
    model.add(layers.SimpleRNN(3))
    
    return model

def build_model2():
    model = Sequential()
    
    model.add(layers.Embedding(256, 100))
    model.add(layers.SimpleRNN(20))
    model.add(layers.Dense(10, 'softmax'))
    
    return model

# create 'Sequential' model, then add layers with correct parameters
    
def main():
    model1 = build_model1()
    print("=" * 20, "첫번째 모델", "=" * 20)
    model1.summary()
    
    print()
    
    model2 = build_model2()
    print("=" * 20, "두번째 모델", "=" * 20)
    model2.summary()

if __name__ == "__main__":
    main()
