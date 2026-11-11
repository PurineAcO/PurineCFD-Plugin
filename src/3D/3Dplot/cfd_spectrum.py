"""Cl 频谱：FFT 幅值谱、PSD 与斯特劳哈尔数。"""

import matplotlib.pyplot as plt
import numpy as np

from cfd_style import C_BLUE, C_GUIDE, C_GRID, C_RED, save_show


def plot_spectrum(case, u_inf):
    if case.time_axis is None or u_inf is None:
        print('time column or far-field velocity unavailable, spectrum and St are skipped')
        return None

    cl_sig = case.urans['cl'].to_numpy(float)
    t_sig = case.urans['time'].to_numpy(float)
    dt_sig = float(np.median(np.diff(t_sig)))
    fs_sig = 1.0 / dt_sig
    n_samp = cl_sig.size

    sig = cl_sig - cl_sig.mean()
    window = np.hanning(n_samp)
    n_fft = 1 << int(np.ceil(np.log2(n_samp * 16)))

    fft_vals = np.fft.rfft(sig * window, n_fft)
    freq = np.fft.rfftfreq(n_fft, dt_sig)
    amp_spectrum = np.abs(fft_vals) * 2.0 / window.sum()

    psd = (np.abs(fft_vals) ** 2) / (fs_sig * np.sum(window ** 2))
    psd[1:-1] *= 2.0

    k_peak = int(np.argmax(amp_spectrum[1:])) + 1
    y0, y1, y2 = np.log(amp_spectrum[k_peak - 1 : k_peak + 2])
    delta = 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2) if np.isfinite(y0 + y2) else 0.0
    f_peak = float(freq[k_peak] + delta * freq[1])
    st_peak = f_peak * case.diam / u_inf

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), layout='constrained')

    axes[0].plot(freq, amp_spectrum, color=C_BLUE, linewidth=1.2)
    axes[0].axvline(f_peak, color=C_GUIDE, linewidth=1.0, linestyle='--')
    axes[0].plot([f_peak], [amp_spectrum[k_peak]], marker='o', color=C_RED, markersize=5.0, linestyle='none')
    axes[0].set_xlabel('f [Hz]')
    axes[0].set_ylabel(r'$|\hat{C}_l|$')
    axes[0].set_title(f'FFT Amplitude Spectrum (St = {st_peak:.3f})')
    axes[0].set_xlim(0.0, 4.0 * f_peak)
    axes[0].grid(True, which='major', color=C_GRID, linewidth=0.6)

    axes[1].semilogy(freq, psd, color=C_RED, linewidth=1.2)
    axes[1].axvline(f_peak, color=C_GUIDE, linewidth=1.0, linestyle='--')
    axes[1].plot([f_peak], [psd[k_peak]], marker='o', color=C_BLUE, markersize=5.0, linestyle='none')
    axes[1].set_xlabel('f [Hz]')
    axes[1].set_ylabel(r'PSD [$1/\mathrm{Hz}$]')
    axes[1].set_title('Power Spectral Density')
    axes[1].set_xlim(0.0, 4.0 * f_peak)
    axes[1].grid(True, which='major', color=C_GRID, linewidth=0.6)

    df_raw = fs_sig / n_samp
    print(f'samples = {n_samp}, dt = {dt_sig:.5f} s, fs = {fs_sig:.2f} Hz')
    print(f'window = {t_sig[0]:.4f} ... {t_sig[-1]:.4f} s, raw df = {df_raw:.3f} Hz')
    print(f'dominant frequency (FFT peak) = {f_peak:.4f} Hz')
    print(f'D = {case.diam:.4f} m, U_inf = {u_inf:.4f} m/s')
    print(f'Strouhal number St = {st_peak:.4f}')
    print(f'uncertainty: df = +/-{0.5 * df_raw:.3f} Hz, dSt = +/-{0.5 * df_raw * case.diam / u_inf:.4f}')

    return save_show(fig, case.fig_dir, 'cl_fft_psd.png')
