'''
encoding texts (tokenize) using one-hot encoding
'''

import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

def set_token(texts):
  tokenizer = Tokenizer()
  tokenizer.fit_on_text(texts)
  return tokenizer
# return tokenizer fits given texts

def text2seq(text, tokenizer):
  return tokenizer.texts_to_sequences([text])[0]
def seq2onehot(seq, num_word):
  return to_categorical(seq, num_classes = num_word+1)


# Example 1. Simple sentences
text1 = "stand on the shoulders of giants"
text2 = "I can stand on mountains"

tokenizer = set_token([text1, text2])

# check the mapping of [word-index]
print("# of words: ", len(tokenizer.word_index)) 
# 9
print("index of words: ", tokenizer.word_index)  
# {'stand': 1, 'on': 2, 'the': 3, 'shoulders': 4, 'of': 5, 'giants': 6, 'i': 7, 'can': 8, 'mountains': 9}

# 1. to sequence
seq = text2seq(text2, tokenizer)
print(seq)
# [7, 8, 1, 2, 9]

# 2. one-hot encoding
onehot1 = seq2onehot(seq, len(tokenizer.word_index))
print(onehot1)
'''
 [[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
'''  # each word has length 10 vector

# Example 2. Long sentences
text3 = "i have copy of this on vhs i think they the television networks should play this every year for the next twenty years so that we don't forget what was and that we remember not to do the same mistakes again like putting some people in the"
text4 = "he old neighborhood in serving time for an all to nice crime of necessity of course john heads back onto the old street and is greeted by kids dogs old ladies and his peer"

tokenizer2 = set_token([text1, text2, text3, text4])
print("# of words: ", len(tokenizer.word_index)) # 69 -> too big! (each word will have length 70 vector)

# one hot encoding for text2 again, using tokenizer2 instead of tokenizer
seq2 = text2seq(text2, tokenizer2)
onehot2 = seq2onehot(seq2, len(tokenizer2.word_index))
print(onehot2)
'''
[[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
''' # each word has length 70 vector, consists of only one '1' -> waste!

# this code shows the inefficiency of one-hot encoidng for natural language data.
