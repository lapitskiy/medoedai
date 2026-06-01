import requests, json

payload = {
    "result_dirs": ["result/xgb/BTC/runs/xgb-02737c"],
    "days_grid": [30],
    "exit_modes": ["policy"],
    "atr_len": 14,
    "atr_mult": 2,
    "p_enter_grid_enabled": False,
    "p_enter_threshold": 0.5,
    "signal_exit_enabled": False
}

try:
    r = requests.post("http://localhost:5050/xgb_oos_batch_async", json=payload, timeout=10)
    print(r.status_code)
    print(json.dumps(r.json(), indent=2))
except Exception as e:
    print(e)
