import cv2
import numpy as np
import glob
import natsort
import open3d as o3d
import time
#####################################################################################################################################
import torch
from utils_depth import post_process_depth, flip_lr, depth_value_to_depth_image, normalize_image, wait_frame_fps, plot_results
from torch.autograd import Variable

from networks.NewCRFDepth import NewCRFDepth
import config
import time
#####################################################################################################################################
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#####################################################################################################################################
ENCODER = config.encoder
MIN_DEPTH_EVAL=config.min_depth_eval
MAX_DEPTH_EVAL=config.max_depth_eval
ENCODER_PATH = config.encoder_path 

CHECKPOINT_PATH = config.checkpoint_path #newcrf

if CHECKPOINT_PATH =='./model/model_kittieigen.ckpt':
    MAX_DEPTH_EVAL=80
#####################################################################################################################################

#####################################################################################################################################
# depth model
model_depth = NewCRFDepth(version=ENCODER, inv_depth=False, max_depth=MAX_DEPTH_EVAL, pretrained=ENCODER_PATH)

model_depth = torch.nn.DataParallel(model_depth)
model_depth.eval()
model_depth.cuda()
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
model_depth.load_state_dict(checkpoint['model'])
#####################################################################################################################################

# Define camera intrinsics for NYU dataset
fx = 5.1885790117450188e+02
fy = 5.1946961112127485e+02
cx = 3.2558244941119034e+02
cy = 2.5373616633400465e+02
scaling_factor = 5000.0

vis = o3d.visualization.Visualizer()
vis.create_window()
opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])
opt.point_size = 1.5

cap = cv2.VideoCapture(0) 

# Used as counter variable 
count = 0

# checks whether frames were extracted 
success = 1

while success:
    with torch.no_grad(): 
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size
        success, image = cap.read() 
        origin_image = image    
    # ###########################################################################################
        image = np.expand_dims(image, axis=0)
        image = np.transpose(image, (0, 3, 1, 2))
        image = Variable(torch.from_numpy(image)).cuda()
        image = image.float()  
        image /= 255.0

        image_flipped = flip_lr(image)
        depth_ests = model_depth(image)
        depth_ests_flipped = model_depth(image_flipped)
        pred_depth = post_process_depth(depth_ests, depth_ests_flipped)
        depth = np.zeros((image.shape[2], image.shape[3]), dtype=np.float32)
        temp = pred_depth[0].cpu().detach().numpy().squeeze()
    
        depth[:, :] = temp / MAX_DEPTH_EVAL     
###########################################################################################
        # Convert depth map to meters
        depth = depth.astype(float) / 1000.0

        # Compute 3D point cloud
        rows, cols = depth.shape[:2]
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        z = depth.astype(np.float32)
        x = np.multiply((c - cx) / fx, z)
        y = np.multiply((r - cy) / fy, z)
        point_cloud = np.stack((x, y, z), axis=-1).reshape(-1, 3)

        # Create Open3D point cloud from NumPy array
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        color_array = np.asarray(origin_image)
        color_array = color_array.reshape((480 * 640, 3))
        pcd.colors = o3d.utility.Vector3dVector(color_array.astype(np.float32) / 255.0)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Visualize point cloud using Open3D
        # o3d.visualization.draw_geometries([pcd])
        cv2.namedWindow('rgb_image', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
        cv2.imshow('rgb_image', origin_image)
        cv2.waitKey(1)
        
        
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(1)
        # vis.run()
        vis.clear_geometries()

    # Close visualizer window
    vis.destroy_window()

