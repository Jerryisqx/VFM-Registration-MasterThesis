from skimage.transform import rotate, rescale, AffineTransform, warp
from scipy.spatial.distance import cosine
from skimage.color import rgb2gray
from skimage.io import imread
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
# make sure renderers.py is in the same directory or in PYTHONPATH
from renderers import simple_renderer
# make sure NFD.py is in the same directory or in PYTHONPATH
from scipy.spatial.transform import Rotation as R
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_tmat(translation=(0, 0, -900), rotation_deg=(0, 0, 0)):
    """
    construct a 4x4 transformation matrix from translation and ZXY Euler angles (in degrees)
    """
    r = R.from_euler('ZXY', rotation_deg, degrees=True)
    rot_mat = r.as_matrix()

    tmat = np.eye(4)
    tmat[:3, :3] = rot_mat
    tmat[:3, 3] = translation
    return torch.tensor(tmat, dtype=torch.float32).unsqueeze(0).to(device)

def test_renderer():
    """
    test renderer's functionality by generating a virtual X-ray image and displaying it
    """
    # model paths (ensure the paths exist)
    femur_path = rf'C:\Users\Public\DPZM_02\postop\gt_03_new\outputdpzm_02\anatomies\DPZM_02_original\ana_000001.stl'
    tibia_path = rf'C:\Users\Public\DPZM_02\postop\gt_03_new\outputdpzm_02\anatomies\DPZM_02_original\ana_000002.stl'

    # initialize renderer
    renderer = simple_renderer(
        femur_path=femur_path,
        tibia_path=None,  # if no tibia model, set to None
        renderer_type='STL',
        output_type='6d',
        normalize_translation=False,
        binary=False,
        load_anatomy=True,
    )

    # default 4x4 transformation matrix (identity matrix)
    identity_tmat = torch.eye(4).float().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    identity_tmat[:3, 3] = torch.tensor([0, 0, -900.0])  # move 10mm towards the camera
    identity_tmat = identity_tmat.unsqueeze(0)  # add batch dimension
    # generate virtual X-ray image
    img_width, img_height = 512, 512
    xray_image = renderer.generate_virtual_xray_STL(identity_tmat, identity_tmat, img_width, img_height)

    # show image
    plt.imshow(xray_image.squeeze(), cmap='gray')
    plt.title("Virtual X-ray (Default Pose)")
    plt.axis('off')
    plt.show()



if __name__ == "__main__":
    test_renderer()
