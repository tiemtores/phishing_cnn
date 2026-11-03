"""
outils_cnn.py - Encodage d'une URL pour le CNN (caractere par caractere).

Partage par l'entrainement (phishing_cnn.py) et l'application Streamlit
(app_phishing_cnn.py) : les deux encodent les URL EXACTEMENT de la meme facon.
"""
import numpy as np

MAX_LEN = 200
ALPHABET = ("abcdefghijklmnopqrstuvwxyz0123456789"
            "-._~:/?#[]@!$&'()*+,;=%")
CAR2NUM = {c: i + 2 for i, c in enumerate(ALPHABET)}     # 0 = vide, 1 = inconnu
VOCAB = len(CAR2NUM) + 2


def encoder(liste_urls, max_len=MAX_LEN):
    """Transforme une liste d'URL en matrice d'entiers (n, max_len)."""
    X = np.zeros((len(liste_urls), max_len), dtype="int32")
    for i, u in enumerate(liste_urls):
        for j, c in enumerate(str(u).lower()[:max_len]):
            X[i, j] = CAR2NUM.get(c, 1)
    return X
