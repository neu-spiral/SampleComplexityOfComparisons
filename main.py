"""
Main file for running sample complexity experiments
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('k', type=int, default=0)
    args = parser.parse_args()
