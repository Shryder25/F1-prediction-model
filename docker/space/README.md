---
title: F1 Race Predictor
emoji: 🏎️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8000
pinned: true
short_description: Lap-by-lap F1 win and podium probabilities, streamed live
---

# F1 Race Predictor

Lap-by-lap win, podium and finishing-position probabilities for a Formula 1 race, streamed
to a browser HUD over Server-Sent Events. Pick a race and press play.

This Space runs the synthetic demo data, so it needs no API access. The code, the data
pipeline (Jolpica + FastF1 into DuckDB and dbt) and the model evaluation are on GitHub:
https://github.com/Shryder25/F1-prediction-model
