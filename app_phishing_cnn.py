"""
app_phishing_cnn.py - Application Streamlit du detecteur de phishing (modele CNN)

Meme application que app_phishing.py ; seul le modele change : un reseau
convolutif (CNN 1D) qui lit l'URL caractere par caractere.

  1. Lancer d'abord l'entrainement (une seule fois) :
         python phishing_cnn.py
     -> cree modele_cnn_phishing.keras et metriques_cnn.json
  2. Lancer l'application :
         streamlit run app_phishing_cnn.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score, roc_curve, auc)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import keras                              # noqa: E402

from outils_url import mesurer_url, extraire_hote   # noqa: E402
from outils_cnn import encoder                      # noqa: E402

st.set_page_config(page_title="Détecteur de phishing",
                   page_icon="phishing.png" if Path("phishing.png").exists() else "🛡️",
                   layout="wide")

FICHIERS = ["modele_cnn_phishing.keras", "metriques_cnn.json"]
manquants = [f for f in FICHIERS if not Path(f).exists()]
if manquants:
    st.error("Fichiers introuvables : " + ", ".join(manquants))
    st.info("Lancez d'abord l'entraînement :  `python phishing_cnn.py`")
    st.stop()


# ---------------------------------------------------------------------------
# Chargement (une seule fois grâce au cache)
# ---------------------------------------------------------------------------
@st.cache_resource
def charger_modele():
    return keras.models.load_model("modele_cnn_phishing.keras")


@st.cache_data
def charger_metriques():
    with open("metriques_cnn.json", encoding="utf-8") as f:
        return json.load(f)


modele = charger_modele()
M = charger_metriques()
y_test = np.array(M["y_test"])
probas = np.array(M["probas"])


def probabilite(url):
    # Le CNN lit directement les caracteres de l'URL ; les 10 mesures ne
    # servent plus qu'a expliquer le verdict a l'utilisateur.
    p = float(modele.predict(encoder([url], M["max_len"]), verbose=0)[0][0])
    return p, pd.Series(mesurer_url(url))


# ---------------------------------------------------------------------------
# Barre latérale : le seuil de décision
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ Détecteur de phishing")
st.sidebar.caption("Réseau de neurones convolutif (CNN 1D, TensorFlow / Keras) "
                   "qui lit l'URL caractère par caractère")
seuil = st.sidebar.slider("Seuil de décision", 0.05, 0.95, 0.50, 0.05,
                          help="Au-dessus du seuil, l'URL est classée "
                               "phishing.")
st.sidebar.markdown(
    "- **Seuil bas** : plus d'attaques détectées, mais plus de sites "
    "légitimes bloqués (faux positifs).\n"
    "- **Seuil haut** : moins de fausses alertes, mais plus d'attaques "
    "manquées (faux négatifs).")
st.sidebar.divider()
st.sidebar.caption(f"Jeu d'apprentissage : {M['nb_apprentissage']} URL · "
                   f"jeu de test : {M['nb_test']} URL (données de "
                   "démonstration)")

st.title("Détection d'URL de phishing")
onglet_url, onglet_perf = st.tabs(["🔎 Analyser une URL",
                                   "📊 Performance du modèle"])

# ===========================================================================
# ONGLET 1 : analyser une URL
# ===========================================================================
with onglet_url:
    with st.form("analyse"):
        url = st.text_input(
            "Adresse à analyser",
            placeholder="https://www.exemple.com/compte",
            value="http://paypal-verify.secure-paypal.xyz/login?cmd=verify")
        envoye = st.form_submit_button("Analyser", type="primary")

    # On garde l'URL en memoire : le resultat reste affiche meme apres un
    # autre clic (bouton de la liste, changement de seuil...).
    if envoye and url.strip():
        st.session_state["url_analysee"] = url.strip()

    if "url_analysee" in st.session_state:
        url = st.session_state["url_analysee"]
        p, mesures = probabilite(url)
        dangereux = p >= seuil

        gauche, droite = st.columns([3, 2])
        with gauche:
            if dangereux:
                st.error(f"### ⚠️ PHISHING PROBABLE\n"
                         f"Probabilité de phishing : **{p:.1%}** "
                         f"(seuil {seuil:.0%})")
            else:
                st.success(f"### ✅ PROBABLEMENT LÉGITIME\n"
                           f"Probabilité de phishing : **{p:.1%}** "
                           f"(seuil {seuil:.0%})")
            st.progress(p, text=f"Score du modèle : {p:.3f}")
            st.caption(f"Nom de domaine analysé : `{extraire_hote(url)}`")

            # Signaux d'alerte lisibles par un utilisateur
            signaux = []
            if mesures["adresse_ip"]:
                signaux.append("Une adresse IP remplace le nom du site.")
            if not mesures["https"]:
                signaux.append("La connexion n'est pas chiffrée (http).")
            if mesures["nb_tirets"] >= 2:
                signaux.append("Plusieurs tirets dans l'adresse "
                               "(imitation de marque possible).")
            if mesures["nb_sous_domaines"] >= 2:
                signaux.append("Nombreux sous-domaines empilés.")
            if mesures["mots_suspects"]:
                signaux.append("Mots typiques des pages de vol d'identifiants "
                               "(login, verify, secure…).")
            if "@" in url:
                signaux.append("Le caractère @ peut masquer la vraie "
                               "destination.")
            if mesures["longueur"] > 75:
                signaux.append("Adresse anormalement longue.")
            st.subheader("Signaux repérés dans l'adresse")
            if signaux:
                for s_ in signaux:
                    st.markdown(f"- {s_}")
            else:
                st.markdown("Aucun signal d'alerte évident.")

        with droite:
            st.subheader("Indicateurs de l'adresse")
            st.caption("Pour information : le CNN ne les utilise pas, il lit "
                       "directement les caractères.")
            st.dataframe(mesures.rename("valeur").to_frame(),
                         width="stretch", height=390)

        st.info("Ce verdict est une aide à la décision. En cas de doute, "
                "n'entrez jamais vos identifiants : accédez au site en "
                "tapant vous-même son adresse, et signalez le message.")

    st.divider()
    st.subheader("Analyser plusieurs URL")
    liste = st.text_area(
        "Une URL par ligne",
        "https://www.amazon.com/compte/aide-77\n"
        "http://192.168.12.44/update/session7712?cmd=update&id=7712\n"
        "https://www.impots.gouv.fr/contact\n"
        "http://paypal.com@192.168.1.5/login")
    if st.button("Analyser la liste"):
        st.session_state["liste_analysee"] = liste
    if "liste_analysee" in st.session_state:
        lignes = [u.strip() for u in st.session_state["liste_analysee"]
                  .splitlines() if u.strip()]
        res = []
        for u in lignes:
            p, _ = probabilite(u)
            res.append({"URL": u, "Probabilité": round(p, 3),
                        "Verdict": "Phishing" if p >= seuil else "Légitime"})
        st.dataframe(pd.DataFrame(res), hide_index=True, width="stretch")

# ===========================================================================
# ONGLET 2 : performance du modèle
# ===========================================================================
with onglet_perf:
    pred = (probas >= seuil).astype(int)
    vn, fp, fn, vp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    st.caption(f"Résultats sur les {len(y_test)} URL du jeu de test, "
               f"jamais vues pendant l'entraînement, avec le seuil {seuil:.2f}.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exactitude", f"{(vp + vn) / len(y_test):.1%}")
    c2.metric("Précision", f"{precision_score(y_test, pred, zero_division=0):.1%}",
              help="Parmi les URL bloquées, part de vrai phishing.")
    c3.metric("Rappel", f"{recall_score(y_test, pred, zero_division=0):.1%}",
              help="Part des attaques effectivement détectées.")
    c4.metric("F1-score", f"{f1_score(y_test, pred, zero_division=0):.3f}")

    g, d = st.columns(2)
    with g:
        st.subheader("Matrice de confusion")
        st.dataframe(pd.DataFrame(
            [[vn, fp], [fn, vp]],
            index=["Vrai : légitime", "Vrai : phishing"],
            columns=["Prédit : légitime", "Prédit : phishing"]),
            width="stretch")
        st.markdown(
            f"- ✅ **{vn}** sites légitimes reconnus · **{vp}** attaques "
            "détectées\n"
            f"- 🟠 **{fp}** faux positifs : sites légitimes bloqués à tort\n"
            f"- 🔴 **{fn}** faux négatifs : attaques non détectées, qui "
            "atteignent l'utilisateur")
    with d:
        st.subheader("Courbe ROC")
        tfp, tvp, _ = roc_curve(y_test, probas)
        st.line_chart(pd.DataFrame({"Taux de vrais positifs": tvp},
                                   index=pd.Index(tfp, name="Taux de faux positifs")),
                      height=260)
        st.caption(f"Aire sous la courbe (AUC) : {auc(tfp, tvp):.3f}")

    st.subheader("Effet du seuil sur les deux types d'erreurs")
    seuils = np.round(np.arange(0.05, 0.96, 0.05), 2)
    effet = pd.DataFrame({
        "Faux positifs": [int(((probas >= s) & (y_test == 0)).sum()) for s in seuils],
        "Faux négatifs": [int(((probas < s) & (y_test == 1)).sum()) for s in seuils],
    }, index=pd.Index(seuils, name="Seuil"))
    st.line_chart(effet, height=260)

    st.subheader("Apprentissage")
    h = pd.DataFrame(M["historique"])
    h.index = pd.RangeIndex(1, len(h) + 1, name="Epoch")
    g2, d2, e2 = st.columns(3)
    g2.line_chart(h[["loss", "val_loss"]].rename(
        columns={"loss": "Perte (apprentissage)", "val_loss": "Perte (validation)"}),
        height=240)
    d2.line_chart(h[["accuracy", "val_accuracy"]].rename(
        columns={"accuracy": "Exactitude (apprentissage)",
                 "val_accuracy": "Exactitude (validation)"}), height=240)
    e2.line_chart(h[["auc", "val_auc"]].rename(
        columns={"auc": "AUC (apprentissage)", "val_auc": "AUC (validation)"}),
        height=240)
    st.caption(f"Arrêt automatique (EarlyStopping) après {len(h)} epochs ; "
               f"poids restaurés à la meilleure epoch "
               f"({int(np.argmin(h['val_loss'])) + 1}).")

    # -----------------------------------------------------------------------
    # Pourquoi ce modèle a été retenu (seuil 0,5, même découpage)
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Modèle retenu : comparaison avec le réseau Dense (10 variables)")
    C, D, R_ = M["cnn"], M["dense_10_variables"], M["cnn_domaines_inconnus"]
    lignes = [("Exactitude", "exactitude", "{:.1%}"), ("Précision", "precision", "{:.1%}"),
              ("Rappel", "rappel", "{:.1%}"), ("F1-score", "f1", "{:.3f}"),
              ("Spécificité", "specificite", "{:.1%}"), ("AUC ROC", "auc_roc", "{:.3f}"),
              ("Faux positifs", "fp", "{:.0f}"), ("Faux négatifs", "fn", "{:.0f}")]
    st.dataframe(pd.DataFrame(
        {"CNN (retenu)": [f.format(C[k]) for _, k, f in lignes],
         "Dense (10 variables)": [f.format(D[k]) for _, k, f in lignes],
         "CNN · domaines jamais vus": [f.format(R_[k]) for _, k, f in lignes]},
        index=[n for n, _, _ in lignes]), width="stretch")
    n_cnn = f"{M['parametres']['cnn']:,}".replace(",", " ")
    st.caption(f"Seuil fixé à 0,5 (indépendant du curseur) · même découpage pour le CNN "
               f"et le Dense · paramètres : CNN {n_cnn} / Dense {M['parametres']['dense']}")
    st.warning(
        f"{M['domaines_partages_test']:.0%} des URL du test ont un nom de domaine "
        "déjà présent dans l'apprentissage : les scores ci-dessus sont optimistes. "
        "La dernière colonne (test sur des domaines jamais vus) est plus proche "
        "d'un usage réel.")
