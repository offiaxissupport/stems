# Diagnostics

Scripts that back individual numbers in `docs/REPORT_2026-10.md`. They read results or
replay short windows; none of them defines how a run behaves, so they sit outside the
code fingerprint (`stems/*.py`, `experiments/*.py`).

| Script | Backs |
|---|---|
| `ev_energy_balance.py` | 6.4: why deferred charging draws less energy for the same deliveries |
| `cap_forecast_variants.py` | 6.7: forecast error, margin and cap exceedance with and without the house storage |
| `ev_rl_requests.py` | 6.7: what each trained arm asks of the chargers against what is executed (needs the saved models, which are not in git) |
| `replay_stored_runs.py` | 9: stored EV-study runs replayed on the current code |
| `observation_timing.py` | which hour the load, solar and hot-water demand observations describe |
