import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--angRes", type=int, default=8, help="angular resolution")
parser.add_argument("--aperture_num", type=int, default=4, help = "aperture coding pattern num")

parser.add_argument('--path_for_LF_data', type=str, default='./LF_data/')
parser.add_argument('--path_for_acquired_data', type=str, default='./Acquired_data/')
parser.add_argument('--path_save', type=str, default='./Dst/')

parser.add_argument('--epoch', type=int, default=600)
parser.add_argument('--device', type=str, default='auto')
parser.add_argument('--num_workers', type=int, default=2, help='num workers of the Data Loader')

# test config
parser.add_argument("--test_threshold", type=float, default=0.30, help="threshold tau to create Event")
parser.add_argument("--test_noise", type=float, default=0.175, help="pixel-level noise")
parser.add_argument("--test_th_noise", type=float, default=0.04, help="pixel-level thresh noise")
parser.add_argument("--eps", type=float, default=0.01, help="eps to create Event")


args = parser.parse_args()