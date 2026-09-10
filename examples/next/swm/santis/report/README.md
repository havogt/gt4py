# Report generator

```
python extract.py RESULTS_DIR        # -> report_data.json (merges every *_{single|multiN}_<job>.jsonl)
python page.py                       # -> santis_gpu_scaling_report.html (inline SVG charts)
```

`parse_roofline.py NCU_DIR` turns the ncu roofline CSVs into `profile/roofline.json` (a copy of the Santis one is `roofline_856172.json`). `build_report.py` holds the chart and table functions, `charts.py` the SVG helpers, `page.py` the
prose. The ncu figures in `build_report.ncu_bars` are transcribed from `profile/ncu_summary.md`.
