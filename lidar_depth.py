import os
import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import torch
import time
import open3d as o3d
from networks.NewCRFDepth import NewCRFDepth

def read_configs(config_path:str)-> dict:
    with open(config_path) as handle:
            config = yaml.safe_load(handle)
    return config


def load_model(checkpoint_path, 
               swin_path,
               custom_model_path, 
               MAX_DEPTH, 
               custom_model):

    model = NewCRFDepth(version='large07', 
                        inv_depth=False, 
                        max_depth=MAX_DEPTH, 
                        pretrained=swin_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')


    model = torch.nn.DataParallel(model)

    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    return model

def create_3d_animation(cap):
    config = read_configs('configs/config_distortion.yaml')

    DIST_cof =np.array (config['DISTORTION_COEFFICIENTS'])
    MAX_DEPTH = config['MAX_DEPTH']
    BENCHMARK = config['BENCHMARK']
    IMAGE_DOWN = config['IMAGE_DOWN']
    BUFFER = config['BUFFER']
    CONFIDENCE_THRESHOLD = config['CONFIDENCE_THRESHOLD']
    IOU_THRESHOLD = config['IOU_THRESHOLD']
    ENCODE_PATH = config['DEPTH']['ENCODER_PATH']
    NEWCRFS_MODEL_PATH = config['DEPTH']['NEWCRFS_CHECKPOINT_PATH']

    # # Define camera intrinsics for NYU dataset
    # fx = 5.1885790117450188e+02
    # fy = 5.1946961112127485e+02
    # cx = 3.2558244941119034e+02
    # cy = 2.5373616633400465e+02
    # scaling_factor = 5000.0

    camera_matrix = np.zeros(shape=(3, 3))
    # camera_matrix[0, 0] = 5.4765313594010649e+02
    # camera_matrix[0, 2] = 3.2516069906172453e+02
    # camera_matrix[1, 1] = 5.4801781476172562e+02
    # camera_matrix[1, 2] = 2.4794113960783835e+02

    camera_matrix[0, 0] = 336.12253659588214
    camera_matrix[0, 2] = 316.0195036680626
    camera_matrix[1, 1] = 338.20444576816595
    camera_matrix[1, 2] = 181.18056479485858

    fx = camera_matrix[0, 0]
    fy =camera_matrix[0, 2] 
    cx = camera_matrix[1, 1]
    cy = camera_matrix[1, 2]
    scaling_factor = 5000.0

    # camera_matrix[2, 2] = 1
    # dist_coeffs = np.array([ 3.7230261423972011e-02, -1.6171708069773008e-01, -3.5260752900266357e-04, 1.7161234226767313e-04, 1.0192711400840315e-01 ])
    # dist_coeffs = np.array([[-0.07836904711573132], [0.03200740386560968], [-0.0006087743276905838], [-0.014223625965419043], [-0.0039604234040442315]])
    # Parameters for a model trained on NYU Depth V2
    # new_camera_matrix = np.zeros(shape=(3, 3))
    # new_camera_matrix[0, 0] = 518.8579
    # new_camera_matrix[0, 2] = 320
    # new_camera_matrix[1, 1] = 518.8579
    # new_camera_matrix[1, 2] = 240
    # new_camera_matrix[2, 2] = 1

    # Get the current file's directory path
    dir_path = os.path.realpath(__file__)

    # Split the directory path and file name
    path, filename = os.path.split(dir_path)

    # Set device to cuda
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Get the parent directory of the current file
    file_path = os.path.dirname(os.path.abspath(__file__))

    # Set the path for the model and encoder
    # Join the parent directory path with the model path
    nyu_path = os.path.join(file_path, NEWCRFS_MODEL_PATH)
    swin_path = os.path.join(file_path, ENCODE_PATH)

    # Load the model
    model = load_model(nyu_path, swin_path, nyu_path, MAX_DEPTH, custom_model = False)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    while True:
        with torch.no_grad():
            # Capture a frame from the webcam
            ret, img = cap.read()

            # Convert the image 
            img = cv2.resize(img, (480,640))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the depth values using a depth sensor
            tensor_image = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2).float().cuda()
            tensor_image = tensor_image / 255.
            depth_est = model(tensor_image)

            depth = depth_est[0, 0, :, :].cpu().detach().numpy()

            rgb_image = o3d.geometry.Image(img)
            depth_map = o3d.geometry.Image(depth)
            
            intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, cx, cy)  # replace fx, fy, cx, cy with actual values
            pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_map, intrinsic)

            color_array = np.asarray(rgb_image)
            color_array = color_array.reshape((480 * 640, 3))
            pcd.colors = o3d.utility.Vector3dVector(color_array.astype(np.float32) / 255.0)

            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            vis.add_geometry(pcd)
            vis.update_geometry(pcd)
            vis.poll_events()
            # time.sleep(1)
            vis.clear_geometries()
            vis.update_renderer()

# Open the webcam

cap = cv2.VideoCapture(2)
create_3d_animation(cap)
# Release the webcam
cap.release()