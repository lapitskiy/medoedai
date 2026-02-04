import os
import sys
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.trade_lab_dataset import TradeLabDatasetConfig, write_symbol_runs_dataset_to_tmp


def main() -> int:
    p = argparse.ArgumentParser(description="Build Trade Lab dataset JSON for a symbol from result/*/runs.")
    p.add_argument("--symbol", required=True, help="Symbol like TON or TONUSDT")
    p.add_argument("--model-type", default="dqn", choices=("dqn", "sac"), help="Model type")
    p.add_argument("--out-dir", default="result/trade_lab_tmp", help="Output directory")
    p.add_argument("--max-pkl-mb", type=int, default=64, help="Max train_result.pkl size to unpickle")
    args = p.parse_args()

    cfg = TradeLabDatasetConfig(model_type=args.model_type, out_dir=args.out_dir, max_pkl_mb=int(args.max_pkl_mb))
    path = write_symbol_runs_dataset_to_tmp(args.symbol, cfg=cfg)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

