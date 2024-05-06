import numpy as np
import pandas as pd
from constants import MAX_KELIME,alfabe,karakter_tablosu

def max_kelime_uzunlugu(df, sutun_ismi):
    """veri setindeki en uzun kelimenin boyutunun hesaplayan fonksiyon"""
    
    return int(df[sutun_ismi].str.len().max())

### Model eğitimi fonksiyonları
def data_hazirlik(csv_url):
    """Veriyi okuyup modele göndermeye hazırlayan fonksiyon"""
    
    data_hazirlik = pd.read_csv('dataset.csv', sep = ';',encoding='utf-8-sig')
    # Turn into lower case
    data_hazirlik.loc[:,'dogru_kelime'] = data_hazirlik.loc[:,'dogru_kelime'].str.lower()
    data_hazirlik.loc[:,'dogru_kelime'] = data_hazirlik.loc[:,'dogru_kelime'].str.lower()
    # Drop the ones that misspelled and correct word is the same
    data_hazirlik = data_hazirlik.drop(data_hazirlik[data_hazirlik['yanlis_kelime'] == data_hazirlik['dogru_kelime']].index)
    #Drop duplicates
    #data_hazirlik = data_hazirlik.drop_duplicates(subset=['yanlis_kelime', 'dogru_kelime'])
    # Reset index
    data_hazirlik=data_hazirlik.reset_index(drop=True)
    return data_hazirlik

def padding(arr):
    for row in range(len(arr)):
        for word in range(MAX_KELIME):
            arr[row][word][0]= 1.
    return arr

#veri setindeki tum veriyi dogru ve yanlis kelime olmak uzere one-hot-encode edilmis hallerini donstopen fonsiyon
def girdiHazirlik(data_hazirlik,MAX_KELIME, alfabe,karakter_tablosu):
    ikili_sonuc_x=np.zeros((len(data_hazirlik),MAX_KELIME,len(alfabe)))
    ikili_sonuc_y=np.zeros((len(data_hazirlik),MAX_KELIME,len(alfabe)))
    for satir_sayisi in range(0, len(data_hazirlik)):
        try:
          ikili_sonuc_x[satir_sayisi] = karakter_tablosu.encode((data_hazirlik['yanlis_kelime'][satir_sayisi]),MAX_KELIME)
          ikili_sonuc_y[satir_sayisi] = karakter_tablosu.encode((data_hazirlik['dogru_kelime'][satir_sayisi]),MAX_KELIME)
        except KeyError as er:
         continue
    return ikili_sonuc_x,ikili_sonuc_y

#etiketin basına 0 ekleyip donstopen fonksiyon
def kaydir(one_hot_encoding_kelime):
    for i in range(len(one_hot_encoding_kelime)-1,-1,-1):
        one_hot_encoding_kelime[i] = one_hot_encoding_kelime[i-1]

    temp =np.zeros(len(alfabe))
    temp[0] = 1
    one_hot_encoding_kelime[0] = temp
    return one_hot_encoding_kelime



### Tahmin fonksiyonları

def tahmin_kelime(kelime,encoder_model,decoder_model):

    states_degeri = encoder_model.predict(kelime)
    ciktiy_kelime = np.zeros((1, 1, len(alfabe)))
    ciktiy_kelime[0, 0, 0] = 1 
    stop = False
    decoded_kelime = list()
    while not stop:
        cikti_tokenleri, h, c = decoder_model.predict([ciktiy_kelime] + states_degeri)
        secilen_harf_indeksi = np.argmax(cikti_tokenleri[0, -1, :])
        decoded_kelime.append(secilen_harf_indeksi)
        
        if (secilen_harf_indeksi==0 or len(decoded_kelime) == MAX_KELIME): #kelimenin sonunda
            stop = True

        ciktiy_kelime = np.zeros((1, 1, len(alfabe)))
        ciktiy_kelime[0, 0, secilen_harf_indeksi] = 1.

        states_degeri = [h, c]

        
    # when loop exists return the output sequence
    return decoded_kelime

def tahmin_kelimeye_cevir(kelime_encode):
    kelime_str=[]
    for i in range(len(kelime_encode)):
        kelime_str.append(karakter_tablosu.index2char[kelime_encode[i]])
    return "".join(kelime_str)

