import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from PIL import Image

from model import *
from utils.utils import *
from utils.utils_datasets import MultiTestSetDataLoader


def main(args):
    ''' Create Dir for Save '''
    _, result_dir = create_dir(args)
    result_dir = result_dir.joinpath(f'from_LF_data')
    result_dir.mkdir(exist_ok=True)

    ''' CPU or Cuda '''
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == 'cuda':
        torch.cuda.set_device(device)

    ''' DATA TEST LOADING '''
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)

    ''' MODEL LOADING '''
    view_point_num = args.angRes * args.angRes
    aperture_mask = ApertureMask(view_point_num, args.aperture_num).to(device)
    reconstruct = Reconstruct(view_point_num, args.aperture_num, 64).to(device)
    esim = EventCam(
        args.aperture_num,
        args.test_threshold,
        args.eps,
        args.test_noise,
        args.test_th_noise,
        device
    ).to(device)

    ''' Load Pre-Trained PTH '''
    ckpt_path_mask = 'model_weights/AcqNet.pth'
    ckpt_path_reconstruct = 'model_weights/RecNet.pth'
    aperture_mask.load_state_dict(torch.load(ckpt_path_mask, map_location=device))
    reconstruct.load_state_dict(torch.load(ckpt_path_reconstruct, map_location=device))

    cudnn.benchmark = (device.type == 'cuda')

    ''' TEST '''
    print('\nStart test...')
    with torch.no_grad():
        idx_epoch = args.epoch

        for index, test_name in enumerate(test_Names):
            test_loader = test_Loaders[index]

            save_dir = result_dir.joinpath(test_name)
            save_dir.mkdir(exist_ok=True)

            test(test_loader, device, aperture_mask, reconstruct, esim, save_dir, args, idx_epoch)


def test(test_loader, device, aperture_mask, reconstruct, esim, save_dir, args, idx_epoch, save_image=True):
    for idx_iter, (data, LF_name) in enumerate(test_loader):
        print("\nScene : ", LF_name[0])
        data = data.to(device)
        _, view_point_num, H, W = data.shape
        data = data.reshape(_, view_point_num, H, W, 1)
        ac_event_images = np.zeros([args.aperture_num - 1, H, W, 1], np.float32)
        re_data = np.zeros([view_point_num, H, W, 1], np.float32)

        with torch.no_grad():
            event_images = torch.zeros(1, args.aperture_num - 1, H, W).to(device)

            temporal_images = aperture_mask(data[..., 0], idx_epoch) / view_point_num

            '''event images'''
            event_images = esim(temporal_images)

            input = event_images / 8.0

            tmp_light_field = reconstruct(input)

            ac_event_images[..., 0] = event_images[0].to("cpu").numpy()
            re_data[..., 0] = tmp_light_field[0].to("cpu").numpy()

            re_data[re_data < 0.0] = 0.0
            re_data[re_data > 1.0] = 1.0

        ''' Save Reconstructed LF '''

        if save_image is True:
            save_dir_ = save_dir.joinpath(LF_name[0])
            save_dir_.mkdir(exist_ok=True)
            re_views_dir = save_dir_.joinpath('re_views')
            re_views_dir.mkdir(exist_ok=True)

        re_light_field = np.zeros([view_point_num, H, W, 1], np.uint8)
        for t in tqdm(range(view_point_num), total=view_point_num, ncols=70):
            re_light_field[t] = np.uint8(re_data[t] * 255.0)

            if save_image is True:
                img_save_re = Image.fromarray(re_light_field[t, ..., 0])
                u = int(t % np.sqrt(view_point_num))
                v = int(t / np.sqrt(view_point_num))
                img_save_re.save(str(re_views_dir) + "/%02d_%02d.png" % (v, u))

        if save_image is True:
            acquired_img_dir = save_dir_.joinpath('acquired_data')
            acquired_img_dir.mkdir(exist_ok=True)

            '''Save event images'''
            for t in range(ac_event_images.shape[0]):
                event_images_save = vis_event(ac_event_images[t], event_max_count=8, cbar=False)
                event_images_save.savefig(
                    str(acquired_img_dir) + "/event_image_%05d.png" % (t),
                    bbox_inches='tight',
                    pad_inches=0
                )

    return


if __name__ == '__main__':
    from option import args

    # Use whatever is set in option.py or command line:
    # cpu, cuda:0, cuda:1, or auto
    if not hasattr(args, "device"):
        args.device = "auto"

    main(args)