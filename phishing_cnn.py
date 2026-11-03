"""
===============================================================================
 DETECTEUR DE PHISHING - VERSION CNN (reseau de neurones convolutif)
-------------------------------------------------------------------------------
 Meme jeu de donnees (urls_phishing_20000.csv), meme nettoyage et meme
 decoupage apprentissage / test que phishing_debutant.py.

 Difference : au lieu de 10 variables calculees a la main, le CNN lit l'URL
 caractere par caractere et apprend lui-meme les motifs suspects.

 Le script :
   1. entraine le CNN (et, pour comparer, le reseau Dense de la version
      debutant sur les 10 variables, avec exactement le meme decoupage) ;
   2. affiche les metriques ;
   3. enregistre les figures dans le dossier figures/ :
        courbes_apprentissage.png  matrice_confusion.png  courbe_roc.png
        courbe_precision_rappel.png  scatter_probabilites.png
        scatter_representation.png
   4. enregistre le modele (modele_cnn_phishing.keras) et metriques_cnn.json.

 Pour l'executer :   python phishing_cnn.py
 (TensorFlow, scikit-learn, pandas et matplotlib sont necessaires)
===============================================================================
"""
import json
import os
import random
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                         # figures enregistrees, sans fenetre
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve, average_precision_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf       # noqa: E402
import keras                  # noqa: E402
from keras import layers      # noqa: E402

from outils_url import mesurer_url, extraire_hote   # noqa: E402

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

FICHIER_CSV = "urls_phishing_20000.csv"
DOSSIER_FIG = "figures"
os.makedirs(DOSSIER_FIG, exist_ok=True)
COUL = {"leg": "#2b6cb0", "phi": "#c0392b", "cnn": "#d9622b", "dense": "#5e8c1f"}

# ============================================================================
# ETAPE 1 - CHARGER ET NETTOYER (identique a phishing_debutant.py)
# ============================================================================
df = pd.read_csv(FICHIER_CSV, usecols=["url", "label"]).dropna()
df["url"] = df["url"].astype(str).str.strip()
df = df[df["url"] != ""]
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df[df["label"].isin([0, 1])].astype({"label": int})
ambigues = df.groupby("url")["label"].nunique()
df = df[~df["url"].isin(ambigues[ambigues > 1].index)]
df = df.drop_duplicates(subset="url").reset_index(drop=True)
print("URL :", len(df), "|", df["label"].value_counts().to_dict())

urls = df["url"].to_numpy()
y = df["label"].to_numpy(dtype="float32")

# ============================================================================
# ETAPE 2 - TRANSFORMER CHAQUE URL EN SUITE DE NOMBRES (un nombre par caractere)
# ----------------------------------------------------------------------------
# Un CNN 1D lit une sequence, comme un signal : chaque caractere recoit un
# numero, puis la couche Embedding le transforme en vecteur appris.
# ============================================================================
MAX_LEN = 200            # 200 caracteres couvrent plus de 90 % des URL du jeu
ALPHABET = ("abcdefghijklmnopqrstuvwxyz0123456789"
            "-._~:/?#[]@!$&'()*+,;=%")
CAR2NUM = {c: i + 2 for i, c in enumerate(ALPHABET)}     # 0 = vide, 1 = inconnu
VOCAB = len(CAR2NUM) + 2


def encoder(liste_urls):
    X = np.zeros((len(liste_urls), MAX_LEN), dtype="int32")
    for i, u in enumerate(liste_urls):
        for j, c in enumerate(str(u).lower()[:MAX_LEN]):
            X[i, j] = CAR2NUM.get(c, 1)
    return X


X_seq = encoder(urls)
X_var = pd.DataFrame([mesurer_url(u) for u in urls]).to_numpy(dtype="float32")   # pour la comparaison

# ============================================================================
# ETAPE 3 - SEPARER APPRENTISSAGE / VALIDATION / TEST (meme graine qu'avant)
# ----------------------------------------------------------------------------
# 80 % apprentissage, 20 % test ; puis 20 % de l'apprentissage pour la
# validation (comme validation_split=0.2, mais tire au hasard et stratifie).
# ============================================================================
idx = np.arange(len(df))
i_train, i_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
i_tr, i_val = train_test_split(i_train, test_size=0.2, random_state=42, stratify=y[i_train])
print(f"Apprentissage : {len(i_tr)} | Validation : {len(i_val)} | Test : {len(i_test)}")

hotes = np.array([extraire_hote(u) for u in urls])
partages = np.isin(hotes[i_test], hotes[i_train]).mean()
print(f"URL de test dont le nom de domaine figure aussi dans l'apprentissage : {partages:.1%}")


# ============================================================================
# ETAPE 4 - CONSTRUIRE LE CNN
# ----------------------------------------------------------------------------
#  Embedding(32) -> 3 branches Conv1D (noyaux 3, 5, 7) -> GlobalMaxPooling
#  -> concatenation -> Dense(64) -> Dropout(0.5) -> Dense(1, sigmoide)
#  Chaque noyau est une "fenetre" glissante de 3, 5 ou 7 caracteres :
#  il apprend a reperer des motifs comme ".ml/", "login", "-secure-"...
# ============================================================================
def construire_cnn():
    entree = keras.Input(shape=(MAX_LEN,), name="url")
    x = layers.Embedding(VOCAB, 32, name="embedding")(entree)
    x = layers.SpatialDropout1D(0.1)(x)
    branches = []
    for k in (3, 5, 7):
        b = layers.Conv1D(64, k, padding="same", activation="relu", name=f"conv{k}_1")(x)
        b = layers.Conv1D(64, k, padding="same", activation="relu", name=f"conv{k}_2")(b)
        branches.append(layers.GlobalMaxPooling1D(name=f"pool{k}")(b))
    x = layers.Concatenate(name="concat")(branches)
    x = layers.Dense(64, activation="relu", name="representation")(x)
    x = layers.Dropout(0.5)(x)
    sortie = layers.Dense(1, activation="sigmoid", name="phishing")(x)
    m = keras.Model(entree, sortie, name="cnn_phishing")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return m


def construire_dense(n_var):
    """Le reseau de la version debutant (10 variables), pour comparer."""
    m = keras.Sequential([keras.Input(shape=(n_var,)),
                          layers.Dense(16, activation="relu"), layers.Dropout(0.2),
                          layers.Dense(8, activation="relu"),
                          layers.Dense(1, activation="sigmoid")], name="dense_10_variables")
    m.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return m


def entrainer(modele, Xtr, ytr, Xval, yval, batch):
    arret = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                          restore_best_weights=True)
    t0 = time.time()
    h = modele.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=100, batch_size=batch,
                   callbacks=[arret], verbose=2)
    return h, time.time() - t0


def metriques(y_vrai, p, seuil=0.5):
    pred = (p >= seuil).astype(int)
    vn, fp, fn, vp = confusion_matrix(y_vrai, pred, labels=[0, 1]).ravel()
    return {"exactitude": accuracy_score(y_vrai, pred),
            "precision": precision_score(y_vrai, pred, zero_division=0),
            "rappel": recall_score(y_vrai, pred, zero_division=0),
            "f1": f1_score(y_vrai, pred, zero_division=0),
            "specificite": vn / (vn + fp),
            "auc_roc": roc_auc_score(y_vrai, p),
            "auc_pr": average_precision_score(y_vrai, p),
            "vn": int(vn), "fp": int(fp), "fn": int(fn), "vp": int(vp)}


# ============================================================================
# ETAPE 5 - ENTRAINER LE CNN
# ============================================================================
cnn = construire_cnn()
cnn.summary()
hist_cnn, duree_cnn = entrainer(cnn, X_seq[i_tr], y[i_tr], X_seq[i_val], y[i_val], batch=64)
p_cnn = cnn.predict(X_seq[i_test], verbose=0).ravel()

# ============================================================================
# ETAPE 6 - LE MODELE DENSE (10 variables) SUR LE MEME DECOUPAGE
# ============================================================================
norm = StandardScaler().fit(X_var[i_tr])
dense = construire_dense(X_var.shape[1])
hist_dense, duree_dense = entrainer(dense, norm.transform(X_var[i_tr]), y[i_tr],
                                    norm.transform(X_var[i_val]), y[i_val], batch=32)
p_dense = dense.predict(norm.transform(X_var[i_test]), verbose=0).ravel()

y_test = y[i_test].astype(int)
M_cnn, M_dense = metriques(y_test, p_cnn), metriques(y_test, p_dense)

print("\n================ RESULTATS SUR LE JEU DE TEST ================")
print(pd.DataFrame({"CNN (caracteres)": M_cnn, "Dense (10 variables)": M_dense}).round(4))
print("\nRapport detaille - CNN :")
print(classification_report(y_test, (p_cnn >= 0.5).astype(int),
                            target_names=["legitime", "phishing"], digits=3))

# ============================================================================
# ETAPE 7 - TEST DE ROBUSTESSE : DECOUPAGE PAR NOM DE DOMAINE
# ----------------------------------------------------------------------------
# Ici, aucun domaine du test n'apparait dans l'apprentissage : le modele doit
# reconnaitre des sites qu'il n'a jamais vus (situation reelle).
# ============================================================================
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
g_train, g_test = next(gss.split(idx, y, groups=hotes))
g_tr, g_val = train_test_split(g_train, test_size=0.2, random_state=42, stratify=y[g_train])
cnn_dom = construire_cnn()
entrainer(cnn_dom, X_seq[g_tr], y[g_tr], X_seq[g_val], y[g_val], batch=64)
M_dom = metriques(y[g_test].astype(int), cnn_dom.predict(X_seq[g_test], verbose=0).ravel())
print("\nCNN - test sur des domaines jamais vus :", {k: round(v, 4) for k, v in M_dom.items()})

# ============================================================================
# ETAPE 8 - FIGURES
# ============================================================================
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# 8.1 Courbes d'apprentissage et de validation
h = hist_cnn.history
ep = np.arange(1, len(h["loss"]) + 1)
meilleure = int(np.argmin(h["val_loss"])) + 1
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for a, (cle, titre) in zip(ax, [("loss", "Perte (entropie croisée)"), ("accuracy", "Exactitude"),
                                ("auc", "AUC")]):
    a.plot(ep, h[cle], "o-", ms=4, color=COUL["leg"], label="apprentissage")
    a.plot(ep, h["val_" + cle], "s-", ms=4, color=COUL["cnn"], label="validation")
    a.axvline(meilleure, ls="--", color="grey", lw=1)
    a.set_title(titre); a.set_xlabel("Epoch"); a.grid(alpha=.3)
ax[0].legend()
ax[0].annotate(f"meilleure epoch : {meilleure}", (meilleure, max(h["val_loss"])),
               xytext=(5, 0), textcoords="offset points", fontsize=9, color="grey")
fig.suptitle("CNN - courbes d'apprentissage et de validation", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{DOSSIER_FIG}/courbes_apprentissage.png", dpi=150); plt.close(fig)

# 8.2 Matrices de confusion (CNN et Dense)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for a, (nom, M) in zip(ax, [("CNN (caractères)", M_cnn), ("Dense (10 variables)", M_dense)]):
    cm = np.array([[M["vn"], M["fp"]], [M["fn"], M["vp"]]])
    a.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.1%})", ha="center", va="center",
                   fontsize=13, color="white" if cm[i, j] > cm.max() / 2 else "black")
    a.set_xticks([0, 1], ["Prédit : légitime", "Prédit : phishing"])
    a.set_yticks([0, 1], ["Vrai : légitime", "Vrai : phishing"])
    a.set_title(f"{nom}\nexactitude {M['exactitude']:.1%}", fontweight="bold")
    for s in a.spines.values(): s.set_visible(False)
fig.suptitle(f"Matrices de confusion - jeu de test ({len(y_test)} URL, seuil 0,5)")
fig.tight_layout(); fig.savefig(f"{DOSSIER_FIG}/matrice_confusion.png", dpi=150); plt.close(fig)

# 8.3 Courbe ROC
fig, a = plt.subplots(figsize=(6.4, 5.6))
for nom, p, c in [("CNN (caractères)", p_cnn, COUL["cnn"]), ("Dense (10 variables)", p_dense, COUL["dense"])]:
    tfp, tvp, _ = roc_curve(y_test, p)
    a.plot(tfp, tvp, color=c, lw=2, label=f"{nom} - AUC = {auc(tfp, tvp):.3f}")
a.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="hasard (AUC = 0,5)")
a.set_xlabel("Taux de faux positifs (1 - spécificité)"); a.set_ylabel("Taux de vrais positifs (rappel)")
a.set_title("Courbe ROC - jeu de test", fontweight="bold"); a.legend(loc="lower right"); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{DOSSIER_FIG}/courbe_roc.png", dpi=150); plt.close(fig)

# 8.4 Courbe précision-rappel
fig, a = plt.subplots(figsize=(6.4, 5.6))
for nom, p, c in [("CNN (caractères)", p_cnn, COUL["cnn"]), ("Dense (10 variables)", p_dense, COUL["dense"])]:
    pr, rp, _ = precision_recall_curve(y_test, p)
    a.plot(rp, pr, color=c, lw=2, label=f"{nom} - AP = {average_precision_score(y_test, p):.3f}")
a.set_xlabel("Rappel"); a.set_ylabel("Précision"); a.set_ylim(0.45, 1.01)
a.set_title("Courbe précision-rappel", fontweight="bold"); a.legend(loc="lower left"); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{DOSSIER_FIG}/courbe_precision_rappel.png", dpi=150); plt.close(fig)

# 8.5 Scatter des probabilités : score du CNN selon la longueur de l'URL
long_test = np.array([len(u) for u in urls[i_test]])
rng = np.random.default_rng(0)
fig, a = plt.subplots(figsize=(9, 5.4))
for lab, nom, c in [(0, "légitime", COUL["leg"]), (1, "phishing", COUL["phi"])]:
    m = y_test == lab
    a.scatter(np.clip(long_test[m], 0, 300), p_cnn[m] + rng.normal(0, .006, m.sum()), s=9, alpha=.45,
              color=c, label=f"vrai : {nom}")
err = (p_cnn >= .5).astype(int) != y_test
a.scatter(np.clip(long_test[err], 0, 300), p_cnn[err], s=40, facecolors="none", edgecolors="black",
          lw=.8, label=f"erreurs ({err.sum()})")
a.axhline(.5, ls="--", color="grey", lw=1); a.text(302, .5, "seuil 0,5", va="center", fontsize=9, color="grey")
a.set_xlabel("Longueur de l'URL (caractères, tronquée à 300)"); a.set_ylabel("Probabilité de phishing (CNN)")
a.set_title("Scatter plot - score du CNN pour chaque URL du jeu de test", fontweight="bold")
a.legend(loc="center right", markerscale=1.5); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{DOSSIER_FIG}/scatter_probabilites.png", dpi=150); plt.close(fig)

# 8.6 Scatter de la représentation apprise (couche Dense 64 -> ACP en 2D)
extracteur = keras.Model(cnn.input, cnn.get_layer("representation").output)
Z = PCA(n_components=2, random_state=0).fit_transform(extracteur.predict(X_seq[i_test], verbose=0))
fig, a = plt.subplots(figsize=(7.5, 6))
for lab, nom, c in [(0, "légitime", COUL["leg"]), (1, "phishing", COUL["phi"])]:
    m = y_test == lab
    a.scatter(Z[m, 0], Z[m, 1], s=8, alpha=.45, color=c, label=nom)
a.scatter(Z[err, 0], Z[err, 1], s=36, facecolors="none", edgecolors="black", lw=.8, label="erreurs")
a.set_xlabel("Composante principale 1"); a.set_ylabel("Composante principale 2")
a.set_title("Ce que le CNN a appris : représentation interne\n(couche de 64 neurones projetée en 2D par ACP)",
            fontweight="bold")
a.legend(markerscale=2); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{DOSSIER_FIG}/scatter_representation.png", dpi=150); plt.close(fig)

# ============================================================================
# ETAPE 9 - ENREGISTRER LE MODELE ET LES RESULTATS
# ============================================================================
cnn.save("modele_cnn_phishing.keras")
with open("metriques_cnn.json", "w", encoding="utf-8") as f:
    json.dump({"cnn": {k: float(v) for k, v in M_cnn.items()},
               "dense_10_variables": {k: float(v) for k, v in M_dense.items()},
               "cnn_domaines_inconnus": {k: float(v) for k, v in M_dom.items()},
               "parametres": {"cnn": int(cnn.count_params()), "dense": int(dense.count_params())},
               "duree_s": {"cnn": round(duree_cnn, 1), "dense": round(duree_dense, 1)},
               "epochs": {"cnn": len(hist_cnn.history["loss"]), "dense": len(hist_dense.history["loss"])},
               "domaines_partages_test": float(partages),
               "y_test": y_test.tolist(), "probas": p_cnn.round(5).tolist(),
               "historique": {k: [float(v) for v in vals] for k, vals in hist_cnn.history.items()},
               "nb_apprentissage": int(len(i_train)), "nb_test": int(len(i_test)),
               "max_len": MAX_LEN, "alphabet": ALPHABET}, f)
print("\nModele : modele_cnn_phishing.keras | Resultats : metriques_cnn.json | Figures :", DOSSIER_FIG)


# ============================================================================
# ETAPE 10 - UTILISER LE CNN SUR DE NOUVELLES URL
# ============================================================================
def analyser(url):
    p = float(cnn.predict(encoder([url]), verbose=0)[0][0])
    return f"{p:6.1%}  {'DANGEREUX' if p >= 0.5 else 'PROBABLEMENT SUR':16}  <-  {url}"


print("\n--- Test sur de nouvelles URL ---")
for essai in ["https://www.amazon.com/compte/aide-77",
              "http://paypal-verify.secure-paypal.xyz/login/session4821?cmd=verify&id=91",
              "https://www.impots.gouv.fr/contact",
              "http://192.168.12.44/update/session7712?cmd=update&id=7712"]:
    print(analyser(essai))
