from flask import Flask, render_template, request
import yanlis_kelime_tespiti_zemberek as zemberek_hazirlik
from functions import tahmin_kelime,tahmin_kelimeye_cevir
from tensorflow.keras.models import load_model # type: ignore
from zemberek import TurkishMorphology,TurkishSentenceNormalizer
from constants import MAX_KELIME,alfabe,karakter_tablosu

morfoloji = TurkishMorphology.create_with_defaults()
norm = TurkishSentenceNormalizer(morfoloji)
encoder_model = load_model('models/new_encoder_model_dataset_tasarim.h5',compile=False)
decoder_model = load_model('models/new_decoder_model_dataset_tasarim.h5',compile=False)


app = Flask(__name__, template_folder='templates')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        metin = request.form.get("metin")
        hatali_kelimeler,liste_metin,hatali_kelime_indeks = zemberek_hazirlik.yanlis_kelime_tespiti(metin,norm)
        print(hatali_kelimeler)
        dogru_kelimeler=[]
        for yanlis_kelime in hatali_kelimeler:
            
            kelime_encode = karakter_tablosu.encode(yanlis_kelime, MAX_KELIME)
            kelime_encode = kelime_encode.reshape(-1,MAX_KELIME,len(alfabe))
            tahmin_kelime_decode = tahmin_kelime(kelime_encode,encoder_model=encoder_model,decoder_model=decoder_model)
            sonuc=tahmin_kelimeye_cevir(tahmin_kelime_decode)
            dogru_kelimeler.append(sonuc)
        for indeks in range(len(hatali_kelime_indeks)):
            liste_metin[hatali_kelime_indeks[indeks]]=dogru_kelimeler[indeks]
            
        return render_template("index.html", h_metin=hatali_kelimeler,d_metin=liste_metin, mtn=metin)
    else:
        return render_template("index.html")




if __name__ == "__main__":
    app.run(debug=False, host='127.0.0.1', port=8000)
