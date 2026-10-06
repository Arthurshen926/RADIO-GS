"""Native-map provider boundary shared by all query modalities."""
import argparse
from .native_field import run_map_query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--queries', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run_map_query(args.scene, args.queries, args.output, device='cuda')


if __name__ == '__main__':
    main()
