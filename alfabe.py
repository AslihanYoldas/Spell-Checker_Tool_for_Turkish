import numpy as np 
class Alfabe(object):
    """Verilen karakterleri siralayip sozluk haline getiren sinif
    """
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char2index = dict((c, i) for i, c in enumerate(self.chars))
        self.index2char = dict((i, c) for i, c in enumerate(self.chars))
        self.size = len(self.chars)
    def get_alfabe(self):
        """Alfabeyi geri donduren fonksiyon"""
        return self.chars
    
    def encode(self, string, nb_rows):
        """Verilen stringin one-hot-encode edilmis halini donduren fonksiyon"""
        encoded_string = np.zeros((nb_rows, len(self.chars)), dtype= np.float32)
            
        for i, c in enumerate(string):
            encoded_string[i, self.char2index[c]] = 1.0
        #paddingi yapıyoruz
        for i in range (len(encoded_string)):
            flag=True
            for j in range(self.size):
                if encoded_string[i][j]==1.:
                    flag=False
            if flag :
                encoded_string[i][0]=1.
            
        return np.asanyarray(encoded_string,dtype=np.float32)

    def decode(self, x, calc_argmax=True):
        """iki boyutlu vektoru one-hot-decode edilmis halini ve string halini donduren fonksiyon"""
        if calc_argmax:
            indices = x.argmax(axis=-1)
        else:
            indices = x
        chars = ''.join(self.index2char[ind] for ind in indices)
        return indices, chars
