import zemberek

def metne_cevir(kelime):
    return ''.join(kelime)

def hatali_mi(kelime,norm):
    try:
        duzeltilen_kelime = norm.normalize(kelime)
        if duzeltilen_kelime == kelime:
            return False #hatali degil
        else:
            return True #hatali
    except IndexError as error:
        return False
    
def yanlis_kelime_tespiti(metin,norm):
    sayac=0
    indeks_yanlis_kelime=[]
    yanlis_kelimeler=[]
    kelime = []
    gurultulu_metin=[]
    for karakter_sayac in range(len(metin)):
        karakter = metin[karakter_sayac]
        if karakter.isalpha():
            karakter=karakter.lower()
            kelime.append(karakter)
        elif len(kelime)>0:
            kelime = metne_cevir(kelime)
            if hatali_mi(kelime,norm):
                yanlis_kelimeler.append(kelime)
                indeks_yanlis_kelime.append(sayac)
            gurultulu_metin.append(kelime)
            sayac+=1
            kelime = []
        if len(metin) == karakter_sayac + 1 : #sondaki kelime icin (bosluk olmadigindan yakalanmiyor yukarida)
            kelime = metne_cevir(kelime)
            if hatali_mi(kelime,norm):
                yanlis_kelimeler.append(kelime)
            gurultulu_metin.append(kelime)
            kelime = []
    return yanlis_kelimeler,gurultulu_metin,indeks_yanlis_kelime

#morfoloji = zemberek.TurkishMorphology.create_with_defaults()
#norm = zemberek.TurkishSentenceNormalizer(morfoloji)
#print(yanlis_kelime_tespiti("merhaba cok güzel", norm))