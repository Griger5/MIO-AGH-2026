# Sprawozdanie: Kompresja danych EKG z uzyciem autoenkodera MLP

Autorzy: ...

## 1. Opis zadania

Celem projektu bylo zbadanie mozliwosci kompresji pojedynczych cykli sygnalu EKG za pomoca sztucznej sieci neuronowej typu MLP. Zastosowano autoenkoder, ktory koduje 187-probkowy sygnal do wektora latentnego o wymiarze `K`, a nastepnie rekonstruuje przebieg wejsciowy.

Badano zaleznosc miedzy stopniem kompresji:

```text
CR = 187 / K
```

a jakoscia odzyskanego przebiegu.

## 2. Dane i preprocessing

Wykorzystano zbior `shayanfazeli/heartbeat` z Kaggle, a konkretnie pliki `mitbih_train.csv` oraz `mitbih_test.csv`. Kazdy rekord zawiera 187 probek pojedynczego cyklu EKG oraz etykiete klasy MIT-BIH.

Wykonane kroki:

- oddzielenie 187 probek sygnalu od etykiety klasy;
- zachowanie normalizacji `[0, 1]` dostarczonej w dataspecie;
- wydzielenie zbioru walidacyjnego ze zbioru treningowego;
- analiza zerowego dopelnienia przez histogram ostatniej niezerowej probki.

## 3. Architektura

Zastosowano symetryczny autoenkoder MLP:

```text
187 -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(K)
K -> Dense(64, ReLU) -> Dense(128, ReLU) -> Dense(187, Sigmoid)
```

Wyjscie `sigmoid` jest zgodne z zakresem danych `[0, 1]`. Funkcja straty to MSE, optymalizator Adam.

Przetestowane wartosci bottlenecka:

```text
K = 2, 4, 8, 16, 32, 64, 96
```

## 4. Metryki

Do oceny zastosowano:

- `PRD` - percentage root-mean-square difference;
- `PRDN` - PRD po odjeciu sredniej sygnalu;
- `SNR` w dB;
- `RMSE`;
- `MAE`;
- `QS = CR / PRD`.

Jako baseline wykorzystano PCA z taka sama liczba skladowych `K`.

## 5. Wyniki

W tej sekcji nalezy wkleic i omowic:

- tabele z `results/metrics.csv`;
- wykres `results/figures/cr_vs_prd.png`;
- wykres `results/figures/cr_vs_qs.png`;
- przykladowe rekonstrukcje `results/figures/reconstructions_k*.png`;
- histogram PRD per klasa `results/figures/prd_by_class_k*.png`;
- histogram dlugosci rzeczywistych sygnalow `results/figures/padding_lengths.png`.

Najwazniejszy wniosek powinien wskazywac, dla ktorego `K` uzyskano najlepszy kompromis miedzy stopniem kompresji a jakoscia rekonstrukcji.

## 6. Dyskusja

Warto omowic:

- czy MLP pokonal PCA przy tym samym `K`;
- przy jakim `CR` sygnal zaczyna tracic istotne cechy ksztaltu;
- czy klasy patologiczne maja gorsza rekonstrukcje niz klasa normalna;
- jaki wplyw moze miec zero-padding;
- ze raportowany `CR` jest teoretyczny, bo nie uwzglednia kwantyzacji latent vectora.

## 7. Zrodla

- Kachuee, M., Fazeli, S., Sarrafzadeh, M. "ECG Heartbeat Classification: A Deep Transferable Representation", 2018.
- Moody, G.B., Mark, R.G. "The impact of the MIT-BIH Arrhythmia Database", 2001.
- Goodfellow, I., Bengio, Y., Courville, A. "Deep Learning", rozdzial o autoenkoderach.
