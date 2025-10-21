# 📊 FFT & Autocorrelation Analysis for Molecular Dynamics

A Python script to **analyze fluctuation modes and relaxation times** from molecular dynamics (MD) simulation data.  
This project computes **FFT magnitudes**, **autocorrelation functions**, and **relaxation times** for per-segment or per-mode data.

---

## ✨ Features

- ✅ Reads **time-series data** from MD simulations (space-separated text files)  
- ✅ Performs **Fast Fourier Transform (FFT)** to convert data from time to frequency domain  
- ✅ Computes **mode amplitudes** as a function of wave vector \(q\)  
- ✅ Calculates **autocorrelation functions** for each mode  
- ✅ Fits autocorrelation to **exponential decay** to extract **relaxation times (\(\tau\))**  
- ✅ Saves results to CSV/TXT files (`fft_magnitudes_all.csv`, `relaxation_times.txt`)  
- ✅ Generates plots:  
  - **Mean FFT magnitude vs wave vector**  
  - **Autocorrelation with exponential fit** per mode  

---

## 🛠️ Dependencies

- `numpy`  
- `pandas`  
- `matplotlib`  
- `scipy`  
- [MDAnalysis](https://www.mdanalysis.org/)  

Install via pip:

```bash
pip install numpy pandas matplotlib scipy MDAnalysis
