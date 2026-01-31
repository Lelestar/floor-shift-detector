# FloorShiftDetector

Computer vision project to detect new/moved objects on the floor by comparing a reference image with a current image.

## Installation
```bash
pip install -r requirements.txt
```

## Usage (pipeline)
1. Open `floorshiftdetector.py` and adjust image paths in `main()`.
2. Run:
```bash
python floorshiftdetector.py
```

## Usage (Streamlit UI)
```bash
streamlit run streamlit_app.py
```
The UI lets you:
- upload images or choose local files,
- tune parameters,
- visualize all pipeline steps,
- export the result.

## Détection du sol (Floor ROI)
Mode disponible :

### `multicluster` (robuste)
Ce mode apprend plusieurs “classes sol” à partir d'une bande basse (parquet + tapis, etc.) :
- **Graines sol** : bande du bas (`floor_seed_ratio`).
- **Features** : couleur LAB + texture.
- **K-means** sur la bande basse → centres “sol”.
- Chaque pixel est gardé s'il est assez proche d'un de ces centres (seuil via quantile).
- Nettoyage morphologique + conservation des régions connectées au bas (optionnel).

#### Comment régler (dans Streamlit)
Dans **Floor ROI** :
- **Floor ROI mode** : choisir `multicluster`.
- **Floor seed ratio (bottom band)** : 0.20–0.35 si le sol contient plusieurs textures (ex: tapis).
- **Number of clusters (K)** : 2–4 (3 par défaut).
- **Seed distance quantile** : 0.85–0.95 (plus haut = masque plus large).
- **Weight a,b / texture** : augmente `texture` si le tapis est mal inclus.
- **Floor mask close/open kernel** : nettoie le masque (ex: close 7 / open 5).
- **Keep only components touching bottom** : utile pour éviter que des zones du haut soient prises pour le sol.

#### Conseils rapides
- Si le tapis n'est pas inclus, augmente `floor_seed_ratio` + `w_tex`.
- Si le masque “mange” les murs, baisse `seed_quantile` ou `w_ab`.
- Si le masque est bruité, augmente légèrement les kernels de nettoyage.

## Project structure
- `floorshiftdetector.py`: main pipeline
- `streamlit_app.py`: web UI
- `Images/`: sample images
- `Sujet - TP1.pdf`: assignment statement
