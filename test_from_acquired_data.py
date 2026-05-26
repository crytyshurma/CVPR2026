import os
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from PIL import Image
import numpy as np

from model import *
from utils.utils import *


def main(args):
    ''' Create Dir for Save '''
    _, result_dir = create_dir(args)
    result_dir = result_dir.joinpath('from_acquired_data')
    result_dir.mkdir(exist_ok=True)

    ''' CPU or Cuda '''
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == 'cuda':
        torch.cuda.set_device(device)

    ''' MODEL LOADING '''
    view_point_num = args.angRes * args.angRes
    reconstruct = Reconstruct(view_point_num, args.aperture_num, 64).to(device)

    ''' Load Pre-Trained PTH '''
    ckpt_path_reconstruct = 'model_weights/RecNet.pth'
    reconstruct.load_state_dict(torch.load(ckpt_path_reconstruct, map_location=device))

    cudnn.benchmark = (device.type == 'cuda')

    ''' TEST '''
    print('\nStart test...')
    with torch.no_grad():

        scenes = [
            name for name in os.listdir(args.path_for_acquired_data)
            if os.path.isdir(os.path.join(args.path_for_acquired_data, name))
        ]

        for scene_name in scenes:
            print('\nScene : ', scene_name)
            save_dir = result_dir.joinpath(scene_name)
            save_dir.mkdir(exist_ok=True)

            H, W = 720, 1280   # EVK4 resolution

            ''' event images '''
            event_images = np.zeros((1, args.aperture_num - 1, H, W))
            for t in range(args.aperture_num - 1):
                event_path = os.path.join(
                    args.path_for_acquired_data,
                    scene_name,
                    "event_image_%05d.npy" % t
                )
                event_images[:, t, :, :] = (
                    np.load(event_path).reshape(1, H, W)
                ).astype('float32')

            event_images = event_images / 8.0
            test_input = torch.from_numpy(event_images).float().to(device)
            test(test_input, device, reconstruct, save_dir, args)


def test(test_input, device, reconstruct, save_dir, args, save_image=True):
    view_point_num = args.angRes * args.angRes
    _, _, H, W = test_input.shape

    with torch.no_grad():
        re_data = np.zeros([view_point_num, H, W, 1], np.float32)

        bias = 5.0   # for brightness correction

        tmp_light_field = bias * reconstruct(test_input)

        re_data[..., 0] = tmp_light_field[0].to("cpu").numpy()
        re_data[re_data < 0.0] = 0.0
        re_data[re_data > 1.0] = 1.0

        ''' Save Reconstructed LF '''
        if save_image is True:
            re_light_field = np.zeros([view_point_num, H, W, 1], np.uint8)
            for t in tqdm(range(view_point_num), total=view_point_num, ncols=70):
                re_light_field[t] = np.uint8(re_data[t] * 255.0)
                img_save_re = Image.fromarray(re_light_field[t, ..., 0])
                u = int(t % np.sqrt(view_point_num))
                v = int(t / np.sqrt(view_point_num))
                img_save_re.save(str(save_dir) + "/%02d_%02d.png" % (v, u))

    return


if __name__ == '__main__':
    from option import args

    if not hasattr(args, "device"):
        args.device = "auto"

    main(args)