# ======= imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from mesh_renderer import MeshRenderer

# ======= constants
th_match = 0.92
square_size = 2.35 # physical size of one square cell in printed checkboard (cm)
feature_extractor = cv2.SIFT_create()
template = cv2.imread("template_rgb.jpg",cv2.IMREAD_UNCHANGED)
video_dir =  "ex_video.mp4"
template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
pixel_to_cm = 2.54 / 96
bfMatch = cv2.BFMatcher()
frames_dir =  "frames/*.jpg"
img_names = glob.glob(frames_dir)
# ======= real world dimensions for scaling
world_im_height = 14
world_im_width = 20
template_height = template.shape[0] * pixel_to_cm
template_width = template.shape[1] * pixel_to_cm
sc_ratio_h = world_im_height / template_height
sc_ratio_w = world_im_height / template_height

# === template image keypoint and descriptors
kp_t, desc_t = feature_extractor.detectAndCompute(template, None)

# ===== video input, output and metadata
def get_frames(v_dir,phys_size = square_size):
    video_frames = []
    cap = cv2.VideoCapture(v_dir)
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(1,totalFrames,5):
        cap.set(cv2.CAP_PROP_POS_FRAMES,i)
        ret, frame = cap.read()
        video_frames.append(frame)

    return video_frames

# Camera calibration
def camera_calib(img_names):
    img_points=[]
    obj_points = []
    num_images = len(img_names)
    pattern_size = (9, 6)
    h, w = cv2.imread(img_names[0]).shape[:2]
    
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    for i, fn in enumerate(img_names):
        print("processing %s... " % fn)
        imgBGR = cv2.imread(fn)

        if imgBGR is None:
            print("Failed to load", fn)
            continue

        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)

        assert w == img.shape[1] and h == img.shape[0], f"size: {img.shape[1]} x {img.shape[0]}"
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        # # if you want to better improve the accuracy... cv2.findChessboardCorners already uses cv2.cornerSubPix
        # if found:
        #     term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        #     cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        if not found:
            print("chessboard not found")
            continue

        # if i < 12:
        #     img_w_corners = cv2.drawChessboardCorners(imgRGB, pattern_size, corners, found)
        #     plt.subplot(3, 4, i + 1)
        #     plt.imshow(img_w_corners)

        print(f"{fn}... OK")
        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

    # plt.show()
    rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    #project_draw(img_names,camera_matrix,dist_coefs,_rvecs, _tvecs)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    return camera_matrix,dist_coefs

def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 4)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 4)

    return img

# AR object points - 4 * square_size = size of 4 chess squares on the chessboard used in calibration. array - pose in 3d space
objectPoints = (
    4
    * square_size
    * np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]])
)
# get camera intrinsics and distoretion coeffs
camera_mat,dist_coef = camera_calib(img_names)
#dist_coef = np.zeros((5,1))
# get video frames
video_frames = get_frames(video_dir)
# create output video
frame_h,frame_w = video_frames[0].shape[0:2]
video_size = (1280,720)
video_output = cv2.VideoWriter('blade_AR_out.avi', 0, 22, video_size)
# ======== create renderer
drill_path = './drill/drill.obj'
sword_path = './sword/blade/scene.gltf'
renderer = MeshRenderer(camera_mat,frame_w, frame_h, sword_path)
# Plotting
(fig,ax) = plt.subplots(1, 1,figsize=(10,10))
fig.suptitle('Warpped Japanese Tsunami', fontsize=16)
ax_data = ax.imshow(np.zeros_like(template))
fig.show()

# ======== Main function
def process_AR_frame(frame):
    # find keypoints feature matches of frame and template
    kp_f, desc_f = feature_extractor.detectAndCompute(frame, None)
    # BFMatch.knnMatch() returns K best matches, where k is specified by the user.
    matches = bfMatch.knnMatch(desc_t,desc_f,k = 2)
    good_and_second_good_match_list = []
    # get best matches with similarity below threshold ratio (the lower the distance, the better)
    """
    Here, knnMatch returns two best matches(k=2), and so since we check m[0]/m[1] < threshold,
    it means m[0] distance is lower than m[1], which is why we end up taking only elments
    on the first row (np.asarray(good_and_second_good_match_list)[:,0]), meaning we take m[0].
    """
    for m in matches:
        if m[0].distance/m[1].distance < th_match:
            good_and_second_good_match_list.append(m)
    good_match_arr = np.asarray(good_and_second_good_match_list)[:,0]
    # ======= Find homography
    good_kp_t = np.array([kp_t[m.queryIdx].pt for m in good_match_arr])
    good_kp_f = np.array([kp_f[m.trainIdx].pt for m in good_match_arr])
    H, masked = cv2.findHomography(good_kp_t, good_kp_f, cv2.RANSAC, 5.0)
    matchesMask = masked.ravel().tolist()
    # ======= take subset of keypoints that obey homography (both frame and reference)
    valid_idx = np.nonzero(matchesMask)
    # fit_kp_t = np.array([kp_t[m.queryIdx].pt for i, m in enumerate(good_match_arr) if masked[i][0] != 0])
    # fit_kp_f = np.array([kp_f[m.trainIdx].pt for i, m in enumerate(good_match_arr) if masked[i][0] != 0])
    fit_kp_f = good_kp_f[valid_idx]
    fit_kp_t = good_kp_t[valid_idx]
    # Adding a zero column, for the Z-axis
    world_kp_t = np.c_[fit_kp_t,np.zeros(fit_kp_t.shape[0])]
    # ======== Pixel to real-world coords conversion
    world_kp_t[:,0] = world_kp_t[:,0] * pixel_to_cm * sc_ratio_w
    world_kp_t[:,1] = world_kp_t[:,1]* pixel_to_cm * sc_ratio_h
    # ======== solve PnP to get cam pose (r_vec and t_vec)
    _,r_vec,t_vec = cv2.solvePnP(world_kp_t,fit_kp_f,camera_mat,dist_coef)
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    renderer.draw(frame_rgb, r_vec, t_vec)
    # ======== project 3D object points onto 2D image-plane, and draw on our frame - for cube
    # imgpts = cv2.projectPoints(objectPoints, r_vec, t_vec, camera_mat, dist_coef)[0]
    # drawn_image = draw(frame, imgpts)

    return frame_rgb

# ====== run on all frames
for i,frame in enumerate(video_frames):
    frame_rgb = process_AR_frame(frame)
    video_frame = cv2.resize(frame_rgb,video_size,interpolation = cv2.INTER_AREA)
    video_frame = cv2.cvtColor(video_frame,cv2.COLOR_RGB2BGR)
    video_output.write(video_frame)
    ax_data.set_data(video_frame)
    fig.canvas.draw()
    fig.canvas.flush_events()

video_output.release()
