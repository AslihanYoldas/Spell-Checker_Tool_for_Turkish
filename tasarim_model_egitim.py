import numpy as np
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Masking # type: ignore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from functions import max_kelime_uzunlugu,girdiHazirlik,kaydir,data_hazirlik
from constants import alfabe,karakter_tablosu



df = data_hazirlik('dataset.csv')


MAX_YANLIS=max_kelime_uzunlugu(df,'yanlis_kelime')
MAX_DOGRU=max_kelime_uzunlugu(df,'dogru_kelime')
MAX_KELIME=max(MAX_YANLIS,MAX_DOGRU)
del MAX_DOGRU 
del MAX_YANLIS


#Modele verilecek girdilerin hazırlanması 
#Girdi degerleri: yanlis kelimenin encoded hali(X),etiketin encoded hali (y)ve etiketin basına 0 eklenmis sekilde encoded hali(sifirli_cikti)
X, y = girdiHazirlik(df,MAX_KELIME,alfabe,karakter_tablosu)
X = np.asarray(X).astype(np.float32)
y = np.asarray(y).astype(np.float32)


del df

X_train, X_test,y_train, y_test = train_test_split(X,y , test_size=0.20, shuffle=False)
del X,y
sifirli_cikti_train = y_train.copy() 
for i, samples in enumerate(sifirli_cikti_train):
    sifirli_cikti_train[i] = kaydir(samples)

sifirli_cikti_test = y_test.copy() 
for i, samples in enumerate(sifirli_cikti_test):
    sifirli_cikti_test[i] = kaydir(samples)

degerler = np.zeros(len(alfabe),dtype=np.float32)
degerler[0] = 1.


# Encoder Model
encode_1= Input(shape=(MAX_KELIME, len(alfabe)), name='encode_1')

masking = Masking(mask_value= degerler)
encoder_girdi_masked = masking(encode_1)

encode_lstm2=LSTM(len(alfabe),return_sequences=True, return_state=True, name='encode_lstm2')
LSTM_ciktilar1, state_h, state_c = encode_lstm2(encoder_girdi_masked)

encoder_states = [state_h, state_c]

# Decoder Model
decoder_girdiler = Input(shape=(None, len(alfabe)), name='decoder_inputs')
decoder_lstm = LSTM(len(alfabe), return_sequences=True, return_state=True, name='decoder_lstm')

decoder_ciktilar,_ , _ = decoder_lstm(decoder_girdiler,
                                     initial_state=encoder_states)

decoder_dense = Dense(len(alfabe), activation='softmax', name='decoder_dense')
decoder_ciktilar = decoder_dense(decoder_ciktilar)

model = Model([encode_1, decoder_girdiler], 
                               decoder_ciktilar, name='model')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
history=model.fit([X_train, sifirli_cikti_train], y_train, batch_size=64, epochs=10, validation_data=([X_test,sifirli_cikti_test],y_test))

# Plot model evaluation
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

encoder_model = Model(encode_1, encoder_states)

decoder_state_girdi_h = Input(shape=(len(alfabe),))
decoder_state_girdi_c = Input(shape=(len(alfabe),))
decoder_states_girdiler = [decoder_state_girdi_h, decoder_state_girdi_c]

decoder_ciktilar, state_h, state_c = decoder_lstm( decoder_girdiler, initial_state=decoder_states_girdiler)
decoder_states = [state_h, state_c]
decoder_ciktilar = decoder_dense(decoder_ciktilar)
decoder_model = Model( [decoder_girdiler] + decoder_states_girdiler,[decoder_ciktilar] + decoder_states) #+ isareti code generate

# Save models
model.save('../models/new_model_dataset_tasarim.h5')
encoder_model.save('../models/new_encoder_model_dataset_tasarim.h5')
decoder_model.save('../models/new_decoder_model_dataset_tasarim.h5')
