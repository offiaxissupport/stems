from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.thermal import (IDX_DAY_TYPE, IDX_HOUR, IDX_T_OUT, IDX_T_OUT_PRED,
                           T_OUT_PRED_LEAD_H, hour_bin, is_weekend,
                           outdoor_temperature_forecast)


def _obs(hour=1, day_type=1, t_out=10.0, t_pred=(16.0, 22.0, 34.0)):
    o = np.zeros(30, dtype=np.float32)
    o[IDX_HOUR], o[IDX_DAY_TYPE], o[IDX_T_OUT] = hour, day_type, t_out
    for k in range(3):
        o[IDX_T_OUT_PRED + k] = t_pred[k]
    return o


def test_hour_bins_are_distinct_for_all_24_hours():
    bins = [hour_bin(_obs(hour=h)) for h in range(1, 25)]
    assert bins == list(range(24))


@pytest.mark.parametrize("day_type, weekend", [
    (1, False), (2, False), (3, False), (4, False), (5, False),
    (6, True), (7, True), (8, True)])
def test_weekend_is_saturday_sunday_and_holidays(day_type, weekend):
    assert is_weekend(_obs(day_type=day_type)) is weekend


def test_forecast_interpolates_the_6_12_24_hour_predictions():
    t = outdoor_temperature_forecast(_obs(), 30)
    assert t[0] == pytest.approx(11.0)
    assert t[5] == pytest.approx(16.0)
    assert t[8] == pytest.approx(19.0)
    assert t[11] == pytest.approx(22.0)
    assert t[23] == pytest.approx(34.0)
    assert t[29] == pytest.approx(34.0)


def test_short_horizon_does_not_see_a_front_six_hours_out():
    t = outdoor_temperature_forecast(_obs(t_out=10.0, t_pred=(-2.0, -2.0, -2.0)), 2)
    assert np.all(t > 5.0)


def test_conventions_hold_on_the_real_travis_data():
    from stems.environment import STEMSEnvironment

    env = STEMSEnvironment(seed=0, heat_pump=True)
    obs, _ = env.reset()
    zeros = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
    rows = [np.asarray(obs[0], dtype=np.float64)]
    for _ in range(72):
        rows.append(np.asarray(env.step(zeros)[0][0], dtype=np.float64))
    X = np.stack(rows)

    hours = X[:, IDX_HOUR].round().astype(int)
    assert hours.min() == 1 and hours.max() == 24
    assert X[0, IDX_DAY_TYPE] == 1, "step 0 is Monday 1 January 2018"

    for k, lead in enumerate(T_OUT_PRED_LEAD_H[:2]):
        lead = int(lead)
        mae = np.abs(X[:-lead, IDX_T_OUT_PRED + k] - X[lead:, IDX_T_OUT]).mean()
        assert mae < 0.5, f"predicted_{k + 1} does not lead by {lead} h (MAE {mae:.2f})"
