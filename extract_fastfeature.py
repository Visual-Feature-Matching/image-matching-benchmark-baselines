import numpy as np
import os
import cv2
from tqdm import tqdm
import argparse
import json
from utils import save_h5
from fastfeature.models.fastfeature_v2 import FastfeatureV2
from vfm.utils.image import image_numpy_to_tensor
import torch


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


def get_SIFT_keypoints(sift, img, nfeatures):
    #! Compute SIFT kpts and desc
    keypoints, desc = sift.detectAndCompute(img, None)

    #! Get response and sort it
    response = np.array([kp.response for kp in keypoints])
    respSort = np.argsort(response)[::-1]

    #! Process kpts
    pt = np.array([kp.pt for kp in keypoints])[respSort]
    size = np.array([kp.size for kp in keypoints])[respSort]
    angle = np.array([kp.angle for kp in keypoints])[respSort]
    response = np.array([kp.response for kp in keypoints])[respSort]


    #! process desc (using RootSIFT here)
    eps = 1e-7
    desc /= (desc.sum(axis=1, keepdims=True) + eps)
    desc = np.sqrt(desc)

    # print(desc[0][:5])
    # print(respSort[0])
    # print(desc[respSort[0]][:5])

    desc = desc[respSort]

    #! Limit nfeatures
    pt = pt[:nfeatures]
    size = size[:nfeatures]
    angle = angle[:nfeatures]
    response = response[:nfeatures]
    desc = desc[:nfeatures]

    # print(desc[0][:5])

    return pt, size, angle, response, desc


if __name__ == '__main__':

    #! Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenes_dir",
        default=os.path.join('..', 'imw-2020'),
        help="path to config file",
        type=str)
    parser.add_argument(
        "--output_dir",
        default=os.path.join('..', 'features_out'),
        type=str)
    parser.add_argument(
        "--lower_sift_threshold",
        default='True',
        type=str2bool,
        help='Lower detection threshold (useful to extract 8k features)')
    parser.add_argument(
        "--subset",
        default='val',
        type=str,
        help='Options: "val", "test", "both", "spc-fix"')
    parser.add_argument(
        "--feature_name",
        type=str)
    parser.add_argument("--nfeatures", default=8000, type=int)

    args = parser.parse_args()

    #! Subset
    if args.subset not in ['val', 'test', 'both', 'spc-fix']:
        raise ValueError('Unknown value for --subset')

    #! Initiaze SIFT
    if args.lower_sift_threshold:
        print('Instantiating SIFT detector with a lower detection threshold')
        sift = cv2.SIFT_create(
            contrastThreshold=-10000, edgeThreshold=-10000)
    else:
        print('Instantiating SIFT detector with default values')
        sift = cv2.SIFT_create()

    #! Initialize Fastfeatures
    ff_config = {
        'device': 'cpu',
        'mode': 'eval',
        'rgb': True,

        'networks':{
            'fastfeature': 'sp-trim',
            'fastfeature_weights': './weights/sptrim128.pth',
            'desc_dim': 128,

            'refine_layer': 'first128',
            'refine_weights': './weights/first128.pth'
        }
    }
    ff_model = FastfeatureV2(ff_config)

    #! Make output dir if not available
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    #! Get list of scenes
    scenes = []
    if args.subset == 'spc-fix':
        scenes += ['st_pauls_cathedral']
    else:
        if args.subset in ['val', 'both']:
            with open(os.path.join('data', 'val.json')) as f:
                scenes += json.load(f)
        if args.subset in ['test', 'both']:
            with open(os.path.join('data', 'test.json')) as f:
                scenes += json.load(f)
    print('Processing the following scenes: {}'.format(scenes))

    #! For each scene
    for scene in scenes:
        print('Processing "{}"'.format(scene))
        kpts_all, desc_all, scores_all, angles_all, scales_all = {}, {}, {}, {}, {}

        scene_path = os.path.join(args.scenes_dir, scene, 'set_100/images/')
        img_list = [x for x in os.listdir(scene_path) if x.endswith('.jpg')]

        #! For each image in the scene
        for im_path in tqdm(img_list):
            img_name = im_path.replace('.jpg', '')

            img = cv2.imread(os.path.join(scene_path, im_path))
            keypoints, scales, angles, responses, desc = get_SIFT_keypoints(
                sift, img, args.nfeatures
            )

            #! Get tensor to feed to model
            img_tensor = image_numpy_to_tensor(img)
            kpts_tensor = torch.as_tensor(keypoints, dtype=torch.float32, device='cpu')
            desc_tensor = torch.as_tensor(desc, dtype=torch.float32, device='cpu')

            #! Feed the model
            full_desc_tensor = ff_model.get_deep_features(img_tensor, kpts_tensor, desc_tensor)

            #! Get results in numpy
            full_desc = full_desc_tensor.detach().cpu().numpy()

            # print(img_name)
            # print(keypoints.shape)
            # print(scales.shape)
            # print(angles.shape)
            # print(responses.shape)
            # print(desc.shape)
            # print('full desc')
            # print(full_desc.shape)

            #! Accumulate results
            kpts_all[img_name] = keypoints
            desc_all[img_name] = full_desc
            scores_all[img_name] = responses
            angles_all[img_name] = angles
            scales_all[img_name] = scales


        #! Make scene output dir if not available
        scene_out_dir = f"{args.output_dir}/{args.feature_name}/{scene}"
        if not os.path.isdir(scene_out_dir):
            os.makedirs(scene_out_dir)

        #! Save H5
        save_h5(kpts_all, f"{scene_out_dir}/keypoints.h5")
        save_h5(desc_all, f"{scene_out_dir}/descriptors.h5")
        save_h5(scores_all, f"{scene_out_dir}/scores.h5")
        save_h5(angles_all, f"{scene_out_dir}/angles.h5")
        save_h5(scales_all, f"{scene_out_dir}/scales.h5")

        print(f"Completed processing scene {scene}")
        input()


    print('Done!')