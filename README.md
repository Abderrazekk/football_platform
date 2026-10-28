# Football Intelligence Platform

Plateforme web intelligente d'analyse et de prédiction des performances
footballistiques (Premier League).

**Projet académique — Tekup | ING-4-J-SDIAF-B | 2026-2027**
**Étudiant :** Abderazek BEL HADJ MBAREK (matricule 25421)

Projet commun validant les matières **Machine Learning** et **Django**.

---

## Description

Application web qui prédit les résultats de matchs de Premier League,
analyse les performances des équipes et des joueurs, et projette la saison
2026-27 (meilleur joueur, meilleur jeune, équipe type, classement final).

**Avertissement :** projet académique d'analyse statistique.
Les prédictions sont probabilistes et ne constituent en aucun cas une
incitation aux paris.

---

## Stack technique

| Couche | Technologie |
|---|---|
| Backend | Python 3.11+, Django |
| Base de données | SQLite |
| Data / ML | pandas, NumPy, scikit-learn, XGBoost, PuLP |
| Frontend | Templates Django, Tailwind CSS, Chart.js / Plotly |
| API (optionnel) | Django REST Framework |

---

## Installation

```bash
# 1. Cloner
git clone <repo-url>
cd football_platform

# 2. Environnement virtuel
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Dépendances
pip install -r requirements.txt

# 4. Migrations
python manage.py migrate

# 5. Lancer
python manage.py runserver