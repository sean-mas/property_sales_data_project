# Milwaukee Property Sales – Analyseprojekt

Dieses Repository enthält das vollständige Analyseprojekt zur
Vorhersage von Immobilienverkaufspreisen in Milwaukee.  Es basiert auf
den öffentlichen `Property Sales`‑Datensätzen der Stadt Milwaukee für die Jahre 2019 – 2024 und dient als Beispiel für ein datenbasiertes
Vorhersageprojekt im Rahmen des Moduls *Predictive Analytics*.

## Inhalte

- **`analysis.ipynb`** – Ausführliches Jupyter‑Notebook mit sämtlichen
  Schritten: Datenbereinigung, explorative Analyse, Feature Engineering,
  Modellbildung (Lineare Regression, Entscheidungsbaum, Random Forest,
  Gradient Boosting), Hyperparameter‑Optimierung sowie Diskussion der
  Ergebnisse.
- **`analysis_script.py`** – Python‑Skript, das die wesentlichen
  Analyseschritte und das Modelltraining ohne Notebook ausführt.
- **Datensätze (`*.csv`)** – Die heruntergeladenen CSV‑Dateien der Jahre
  2019 bis 2024.  Quelle: [Milwaukee Open Data Portal – Property Sales Data](https://data.milwaukee.gov/dataset/property-sales-data)【366478921414254†L38-L154】.
- **`requirements.txt` / `uv.lock`** – Abhängigkeitslisten aus dem
  Modulbeispiel, die hier wiederverwendet werden können.  Für dieses
  Projekt werden primär `pandas`, `numpy`, `scikit‑learn`,
  `matplotlib` und `seaborn` benötigt.

## Einrichtung

1. **Conda‑Umgebung erstellen (empfohlen)**

   ```bash
   conda create -n property-sales python=3.10 -y
   conda activate property-sales
   pip install -r requirements.txt
   ```

   Alternativ kann das Projekt auch mit `uv` oder `pip` betrieben
   werden; alle benötigten Bibliotheken sind in `requirements.txt`
   aufgeführt.

2. **Notebook ausführen**

   ```bash
   jupyter notebook analysis.ipynb
   ```

   oder im Batch‑Modus:

   ```bash
   jupyter nbconvert --to notebook --execute analysis.ipynb --output executed.ipynb
   ```

3. **Skript ausführen (ohne Notebook)**

   ```bash
   python analysis_script.py
   ```

   Dieses Skript lädt automatisch alle CSV‑Dateien im Projektverzeichnis,
   führt die Bereinigung durch, trainiert die Modelle und gibt die
   wichtigsten Kennzahlen sowie den durchschnittlichen Preis vor und
   während der Pandemie aus.

## Reproduzierbarkeit und Erweiterbarkeit

- Alle Transformationsschritte sind in Scikit‑Learn‑Pipelines gekapselt,
  sodass sich neue Modelle oder zusätzliche Features leicht integrieren
  lassen.
- Das Notebook folgt der gleichen Gliederung wie in den Vorlesungen:
  Daten laden, explorieren, transformieren, Modelle vergleichen,
  Ergebnisse interpretieren.
- Zur Erweiterung können weitere Jahre hinzugefügt werden, zusätzliche
  Merkmale (z.B. Nachbarschaft, Gebäudestil) in die Pipelines
  aufgenommen oder alternative Regressionsverfahren (z.B. XGBoost) getestet
  werden.

## Hinweis zum Einfluss der COVID‑19‑Pandemie

Das Projekt untersucht auch, ob die COVID‑19‑Pandemie ab Februar 2020
den Immobilienmarkt beeinflusst hat.  In der Analyse wird ein
binäres Merkmal `Pandemic` verwendet, das Verkäufe ab dem
1. Februar 2020 markiert.  Der durchschnittliche Verkaufspreis sank in
der untersuchten Periode leicht von rund 269 000 USD vor der Pandemie
auf etwa 257 000 USD während der Pandemie.
