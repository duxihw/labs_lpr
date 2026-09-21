# 3.6.1 — Спектральный анализ электрических сигналов

Заготовка работы: отчёт, скрипты обработки и пустые таблицы под данные.
Структура повторяет остальные работы репозитория `labs_lpr/`.

```
3.6.1/
├── data/
│   ├── raw/          сырые данные — заполняются вручную во время работы
│   └── processed/    результат обработки — пишется скриптом, вручную не править
├── report/
│   ├── main.tex          отчёт
│   ├── preamble-local.tex резервная преамбула
│   └── img/              скриншоты осциллографа и графики
└── scripts/
    ├── lab_utils.py      общие утилиты (пути, МНК, округление)
    ├── process_data.py   обработка сырых данных
    └── plot_data.py      графики в «латеховском» стиле
```

## Порядок работы

1. Во время лабораторной заполнить таблицы в `data/raw/` (строки с `#` —
   комментарии с описанием столбцов, их можно не трогать). Сырые данные
   никогда не перезаписываются скриптами.
2. Обработка:

   ```bash
   cd 3.6.1
   python scripts/process_data.py   # -> data/processed/*.csv + сводка в консоль
   python scripts/plot_data.py      # -> report/img/*.pdf
   ```

   `process_data.py` печатает все промежуточные и итоговые числа
   (наклоны МНК с погрешностями, `tau_RC`, произведения `Δν·τ`) и дублирует
   их в `data/processed/fit_results.csv` — оттуда числа переносятся в отчёт
   вместо меток `???`.
3. Скриншоты осциллографа сложить в `report/img/` под именами, которые
   ожидает `main.tex`:
   `signal_rect.png`, `spectrum_rect.png`, `spectrum_rect_1..4.png`,
   `spectrum_burst_1..2.png`, `spectrum_gauss_1..2.png`,
   `filter_signal_1..2.png`.
   Пока файла нет, на его месте в PDF рисуется рамка-заглушка — отчёт
   собирается в любом случае.
4. Сборка отчёта:

   ```bash
   cd report && latexmk -pdf main.tex
   ```

## Заметки по `main.tex`

- Преамбула берётся из `../../common/latex_template/preamble.tex`;
  если файла нет (сборка каталога отдельно от репозитория), подключается
  `preamble-local.tex` — копия той же преамбулы.
- Таблицы и графики читаются напрямую из `../data/processed/*.csv`
  (`csvsimple-l3` + `pgfplots`), поэтому после перезапуска обработки
  достаточно пересобрать PDF.
- Метки `???` и `?? НЕОБХОДИМО ЗАПОЛНИТЬ ??` — места под числа и выводы.
- В графиках pgfplots коэффициенты аппроксимирующих прямых пока заданы
  теоретическими значениями (`1.0*x`, `0.5*x`, `tau_RC = 0.010` мс);
  рядом стоит комментарий `% ???` — подставить значения из МНК.
- При первой сборке стоит просмотреть лог: часть подписей содержит
  кириллицу внутри математического режима (`$a_{\text{бок}}$`), и при
  другой преамбуле может потребоваться `\text{}`/`\mathrm{}`.

## Форматы сырых таблиц

| файл | пункт задания | столбцы |
|---|---|---|
| `setup_params.csv` | — | `param,value,sigma,unit,comment` |
| `harmonics.csv` | 7 | `n,nu_exp_kHz,sigma_nu_kHz,a_exp_mV,sigma_a_mV` |
| `spectrum_width.csv` | 8 | `tau_us,sigma_tau_us,Dnu_kHz,sigma_Dnu_kHz` |
| `harmonics_spacing.csv` | 9 | `T_ms,sigma_T_ms,m,nu_n_kHz,nu_nm_kHz,sigma_nu_kHz` |
| `bursts.csv` | 13–14 | `nu0_kHz,T_ms,N,nu_center_kHz,...,Dnu_kHz,...,dnu_kHz,...` |
| `gauss.csv` | 17 | `nu0_kHz,T_ms,tau_ms,sigma_tau_ms,Dnu_kHz,sigma_Dnu_kHz` |
| `am_modulation.csv` | 20–23 | `m_set_percent,A_max_V,A_min_V,sigma_A_V,a_main_mV,a_side_mV,sigma_a_mV` |
| `pm_modulation.csv` | 24–25 | `phi_m_deg,nu0_kHz,nu_mod_kHz,a_main_mV,a_side1_mV,a_side2_mV,sigma_a_mV` |
| `filter.csv` | 28–29 | `n,nu_kHz,sigma_nu_kHz,a_filt_mV,a_orig_mV,sigma_a_mV` |

Единицы подобраны так, чтобы формулы были «безразмерными»:
кГц × мс = 1, кГц × мкс = 10⁻³.
