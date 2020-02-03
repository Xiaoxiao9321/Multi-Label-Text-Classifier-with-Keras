import argparse
import zipfile
import pandas as pd
import re
import sklearn.metrics
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.layers.convolutional import Conv1D
from keras.layers import GRU, Dropout, Dense
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from numpy import asarray
from numpy import zeros
from keras.models import load_model


emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love",
            "optimism", "pessimism", "sadness", "surprise", "trust"]
emotion_to_int = {"0": 0, "1": 1, "NONE": -1}


def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


def cnnmodel ():
    inputs = Input(shape=(None,))
    embedded = Embedding(vocab_length, 100, weights=[embedding_matrix], trainable=True)(inputs)
    dropout1 = Dropout(0.2)(embedded)
    gru1 = GRU(64, dropout=0.2, activation='tanh', return_sequences=True)(dropout1)
    cov1 = Conv1D(80, 2, activation='relu')(gru1)
    cov2 = Conv1D(80, 2, activation='relu')(cov1)
    # pool1 = MaxPooling1D(4)(cov2)
    pool1= GlobalMaxPooling1D()(cov2)
    #gru1 = GRU(64, dropout=0.2, activation='tanh')(pool1)
    hidden1 = Dense(64, activation='relu')(pool1)
    output = Dense(11, activation='sigmoid')(hidden1)
    model = Model(inputs=inputs, outputs=output)
    opt = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

    return model




if __name__ == "__main__":

    seed = 7
    np.random.seed(seed)

    # gets the training and test file names from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("train", nargs='?', default="2018-E-c-En-train.txt")
    parser.add_argument("test", nargs='?', default="2018-E-c-En-test.txt")
    args = parser.parse_args()

    # reads train and dev data into Pandas data frames
    read_csv_kwargs = dict(sep="\t",
                           converters={e: emotion_to_int.get for e in emotions})
    train_data = pd.read_csv(args.train, **read_csv_kwargs)
    test_data = pd.read_csv(args.test, **read_csv_kwargs)

    # process data to obtain train data and label
    train_sentences = list(train_data['Tweet'])
    train_sent =[preprocess_text(sen) for sen in train_sentences]
    train_y = train_data[emotions].values

    # process data to obtain test data and label
    test_sentence = list(test_data['Tweet'])
    test_sent = [preprocess_text(sen) for sen in test_sentence]
    test_y = test_data[emotions].values

    print(test_y)

    # transfer train words to numeric form
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_sent)
    train_x = tokenizer.texts_to_sequences(train_sent)
    vocab_length = len(tokenizer.word_index) + 1

    # find the train longest length of sentence
    word_count = lambda sentence: len(word_tokenize(sentence))
    longest_sentence = max(train_sent, key=word_count)
    length_long_sentence = len(word_tokenize(longest_sentence))

    # pad train sentences
    train_x = pad_sequences(train_x, length_long_sentence, padding='post')

    # transfer test words to numeric form
    test_tokenizer = Tokenizer()
    test_tokenizer.fit_on_texts(test_sent)
    test_x = tokenizer.texts_to_sequences(test_sent)

    # find the test longest length of sentence
    word_count = lambda sentence: len(word_tokenize(sentence))
    test_longest_sentence = max(test_sent, key=word_count)
    length_test_long_sentence = len(word_tokenize(test_longest_sentence))

    # pad test sentence
    test_x = pad_sequences(test_x, length_test_long_sentence, padding='post')

    # load the GloVe word embedding
    embeddings_dictionary = dict()
    glove_file = open('glove.6B.100d.txt', encoding="utf8")
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()

    # build embedding matrix for my own words
    embedding_matrix = zeros((vocab_length, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    # build a neural network
    model = cnnmodel()
    # early stop
    es = EarlyStopping(monitor='val_acc', verbose=0, mode='max', patience=25, min_delta=0.005,
                       restore_best_weights=True)
    # checkpoint
    filepath = "best_model_4.hdf5"
    cp = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # model fit
    history = model.fit(train_x, train_y, epochs=100, batch_size=32, validation_split=0.05, callbacks=[es,cp])

    # load a saved model
    saved_model = load_model('best_model_4.hdf5')

    #evaluation
    loss, accuracy = saved_model.evaluate(train_x, train_y,batch_size=32)
    print("loss",loss)
    print('accuracy',accuracy)

    # make predictions
    predict_y = saved_model.predict(test_x)
    predict_y = (predict_y > 0.5).astype('int')
    print(predict_y)

    print("accuracy: {:.3f}".format(sklearn.metrics.jaccard_similarity_score(test_y, predict_y)))

    # make zipfile
    dev_predictions = test_data.copy()
    dev_predictions[emotions] = predict_y

    dev_predictions.to_csv("E-C_en_pred.txt", sep="\t", index=False)

    with zipfile.ZipFile('submission.zip', mode='w') as submission_zip:
        submission_zip.write("E-C_en_pred.txt")



    # best_model: 0.529
    # best_model_1 : 0.519
    # best_model_2: 0.508
    # best_model_3: 0.508
    # best_model_4 : 0.538
    # best_model_5 : 0.531






