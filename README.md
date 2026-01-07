# Predictor DS Agent (Data Scientist)

Un agent "data scientist" prêt à l'emploi :
- Analyse automatique d’un CSV (`/analyze`)
- Entraînement automatique (classification ou régression) (`/train`)
- Prédiction sur une nouvelle ligne (`/predict`)

## 1) Prérequis
- Python 3.10+ recommandé
- pip

## 2) Installation (local)
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
source .venv/bin/activate

pip install -r requirements.txt
