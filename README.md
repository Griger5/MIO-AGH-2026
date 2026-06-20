# Kompresja sygnalu EKG z uzyciem autoenkodera MLP

Projekt do kursu MIO: kompresja danych EKG za pomoca sztucznej sieci neuronowej typu MLP.

Autorzy: wpiszcie sklad zespolu przed oddaniem projektu.

## Cel

Celem jest nauczenie autoenkodera, ktory kompresuje pojedynczy cykl EKG z 187 probek do wektora latentnego o wymiarze `K`, a nastepnie rekonstruuje sygnal. Eksperyment sprawdza zaleznosc miedzy stopniem kompresji `CR = 187 / K` a jakoscia rekonstrukcji.

Kod porownuje autoenkoder MLP z bazowym PCA dla tych samych wartosci `K`.

## Dane

Uzywany dataset: Kaggle `shayanfazeli/heartbeat`.

Pobierz pliki przez Makefile:

```bash
make download-data
```

Target uzywa Kaggle CLI, wiec przed uruchomieniem skonfiguruj token Kaggle w `~/.kaggle/kaggle.json` albo ustaw zmienne `KAGGLE_USERNAME` i `KAGGLE_KEY`.

Docelowo powinny powstac pliki:

```text
data/raw/mitbih_train.csv
data/raw/mitbih_test.csv
```

Pliki CSV nie sa commitowane do repozytorium.

Kazdy wiersz MIT-BIH zawiera 187 probek sygnalu oraz etykiete klasy. Dane sa juz znormalizowane do zakresu `[0, 1]` i dopelnione zerami do stalej dlugosci.

## Instalacja

Projekt uzywa `uv` i Pythona 3.12:

```bash
uv sync --python 3.12
```

Alternatywnie:

```bash
make sync
```

## Trening

Pelny eksperyment:

```bash
uv run ecg-train --data-dir data/raw --results-dir results --bottlenecks 2 4 8 16 32 64 96
```

Albo:

```bash
make train
```

`make train` domyslnie wymusza GPU przez `DEVICE=cuda`. Jesli chcesz automatyczny fallback do CPU, uruchom:

```bash
make train DEVICE=auto
```

Domyslne ustawienia Makefile sa dobrane pod GPU:

```text
BATCH_SIZE=2048
NUM_WORKERS=0
HIDDEN_DIMS=2048 1024 512 256
AMP=1
DEVICE=cuda
```

Mozna je zmienic przy uruchomieniu, np.:

```bash
make train BATCH_SIZE=4096 HIDDEN_DIMS="4096 2048 1024 512" AMP=1
```

Szybki test na ograniczonej liczbie wierszy:

```bash
uv run ecg-train --data-dir data/raw --results-dir results --bottlenecks 8 --epochs 2 --limit 512 --device cuda --amp --hidden-dims 2048 1024 512 256
```

Model MLP:

```text
187 -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(K)
K -> Dense(64, ReLU) -> Dense(128, ReLU) -> Dense(187, Sigmoid)
```

## Ewaluacja

```bash
uv run ecg-evaluate --data-dir data/raw --results-dir results
```

Albo:

```bash
make evaluate
```

Wyniki:

```text
results/metrics.csv
results/figures/cr_vs_prd.png
results/figures/cr_vs_qs.png
results/figures/cr_vs_rmse.png
results/figures/cr_vs_snr.png
results/figures/padding_lengths.png
results/figures/reconstructions_k*.png
results/figures/prd_by_class_k*.png
```

Liczone metryki:

- `CR = 187 / K`
- `PRD`
- `PRDN`
- `SNR`
- `RMSE`
- `MAE`
- `QS = CR / PRD`

## Demo i inferencja

Demo uzywa najlepszego juz wytrenowanego modelu MLP wedlug `results/metrics.csv` i zapisuje przykladowe rekonstrukcje:

```bash
make demo
```

Domyslnie wybierany jest najnizszy `PRD`. Inna metryka:

```bash
make demo BEST_METRIC=rmse
```

Inferencja dla gotowego checkpointu:

```bash
make infer INFER_K=8 INFER_SAMPLES=6
```

Albo dla konkretnych indeksow ze zbioru testowego:

```bash
make infer INFER_K=8 INFER_INDICES="0 10 20 30"
```

Wyniki inferencji:

```text
results/inference_k8/reconstructions.csv
results/inference_k8/latent.csv
results/inference_k8/sample_metrics.csv
results/inference_k8/metrics.json
results/inference_k8/reconstructions_k8.png
```

Szybki demonstracyjny trening od zera jest oddzielnym targetem:

```bash
make quick-demo
```

## Testy

```bash
uv run pytest
```

Albo:

```bash
make test
```

Testy uzywaja syntetycznych sygnalow EKG, wiec nie wymagaja datasetu Kaggle.

## Material do sprawozdania

W sprawozdaniu warto opisac:

- format danych MIT-BIH z Kaggle;
- preprocessing: oddzielenie etykiety, split walidacyjny, analiza zero-paddingu;
- architekture autoenkodera MLP;
- badane rozmiary bottlenecka `K`;
- porownanie MLP z PCA;
- wykresy `CR vs PRD` i `CR vs QS`;
- przykladowe rekonstrukcje sygnalu;
- obserwacje per klasa MIT-BIH.

Podstawowe zrodla do cytowania:

- Kachuee, M., Fazeli, S., Sarrafzadeh, M. "ECG Heartbeat Classification: A Deep Transferable Representation", 2018.
- Moody, G.B., Mark, R.G. "The impact of the MIT-BIH Arrhythmia Database", 2001.
- Goodfellow, Bengio, Courville, "Deep Learning", rozdzial o autoenkoderach.
