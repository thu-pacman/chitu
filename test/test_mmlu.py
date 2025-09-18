import argparse
import json

from evalscope.run import run_task

parser = argparse.ArgumentParser(description="Run evalscope evaluation with config")
parser.add_argument("--config", type=str)
args = parser.parse_args()
if args.config:
    task_cfg = json.loads(args.config)

run_task(task_cfg=task_cfg)
