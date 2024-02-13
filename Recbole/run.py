import argparse
import wandb
from recbole.quick_start import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="level3", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default="recbole/properties/overall.yaml", help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )

    args, _ = parser.parse_known_args()

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )
    run(
        args.model,
        args.dataset,
        config_file_list=config_file_list,
        nproc=args.nproc,
        world_size=args.world_size,
        ip=args.ip,
        port=args.port,
        group_offset=args.group_offset,
    )