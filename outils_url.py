"""
outils_url.py - Calcul des 10 variables explicatives d'une URL.

Ce module est partage par le script d'entrainement (phishing_debutant.py)
et par l'application Streamlit (app_phishing.py) : les deux calculent ainsi
les variables EXACTEMENT de la meme facon.
"""
import ipaddress
from urllib.parse import urlsplit

MOTS_SUSPECTS = ["login", "secure", "verify", "update", "confirm",
                 "account", "banque", "password"]


def extraire_hote(url):
    """Renvoie le nom d'hote ("www.site.com"), sans port ni identifiant."""
    # CORRECTION : urlsplit gere correctement "user@site.com", ":8080",
    # les majuscules et les "//" situes plus loin dans l'URL.
    if "//" not in url.split("?")[0][:12]:
        url = "http://" + url                    # URL saisie sans schema
    return (urlsplit(url).hostname or "").lower()


def est_adresse_ip(hote):
    try:
        ipaddress.ip_address(hote)
        return 1
    except ValueError:
        return 0


def mesurer_url(url):
    hote = extraire_hote(url)
    ip = est_adresse_ip(hote)
    sans_www = hote[4:] if hote.startswith("www.") else hote
    return {
        "longueur": len(url),                                   # 1
        "longueur_hote": len(hote),                             # 2
        "nb_points": url.count("."),                            # 3
        "nb_tirets": url.count("-"),                            # 4
        "nb_chiffres": sum(c.isdigit() for c in url),           # 5
        "nb_speciaux": sum(c in "@?&=%" for c in url),          # 6
        "https": int(url.lower().startswith("https://")),       # 7
        "adresse_ip": ip,                                       # 8
        # CORRECTION : une adresse IP n'a pas de sous-domaines, et "www"
        # n'est pas compte comme sous-domaine.
        "nb_sous_domaines": 0 if ip else max(sans_www.count(".") - 1, 0),  # 9
        "mots_suspects": sum(m in url.lower() for m in MOTS_SUSPECTS),  # 10
    }
