import cv2
import numpy as np
import glob
import natsort
import open3d as o3d
import time

img_path_list=natsort.natsorted(glob.glob('/home/nam/바탕화면/sadat/rgb/*.jpg'))  
depth_path_list=natsort.natsorted(glob.glob('/home/nam/바탕화면/sadat/depth/*.png'))  

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


for index, (img_path, depth_path) in enumerate(zip(img_path_list, depth_path_list)):
    print(img_path)
    # Load RGB image and depth map
    rgb = cv2.imread(img_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

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

    color_array = np.asarray(rgb)
    color_array = color_array.reshape((480 * 640, 3))
    pcd.colors = o3d.utility.Vector3dVector(color_array.astype(np.float32) / 255.0)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Visualize point cloud using Open3D
    # o3d.visualization.draw_geometries([pcd])
    cv2.namedWindow('rgb_image', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
    cv2.imshow('rgb_image', rgb)
    cv2.waitKey(1)

    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    # vis.capture_screen_image(f'./results/save_{index}.png')

    time.sleep(1)
    # vis.run()
    vis.clear_geometries()

# Close visualizer window
vis.destroy_window()

