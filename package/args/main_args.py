import argparse


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--start_from', type=str, default=None)
    parser.add_argument('--single', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='test')
    return parser.parse_args()


if __name__ == '__main__':
    pass



