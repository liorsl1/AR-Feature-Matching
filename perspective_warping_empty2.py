# ======= imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ======= constants
feature_extractor = cv2.SIFT_create()
template = cv2.imread("template_rgb.jpg")
#template = cv2.cvtColor(template,cv2.COLOR_BGR2RGB)
video_dir =  "template_video2.mp4"
bfMatch = cv2.BFMatcher()
# === plot
# (fig,ax) = plt.subplots(1, 1,figsize=(11,11))
# ax_data = ax.imshow(np.zeros_like(template))
# fig.suptitle('Warpped Japanese Tsunami', fontsize=16)
#fig.show()
# === template image keypoint and descriptors
kp_t, desc_t = feature_extractor.detectAndCompute(template, None)

# ===== video input, output and metadata
def get_frames(v_dir):
    video_frames = []
    cap = cv2.VideoCapture(v_dir)
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(1,totalFrames,5):
        cap.set(cv2.CAP_PROP_POS_FRAMES,i)
        ret, frame = cap.read()
        video_frames.append(frame)

    return video_frames

def warpTwoImages(img1, img2, H):
    '''
    (The original function taken from the SIFT notebook)
    warp img2 to img1 with homograph H
    from: https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective
    
    '''
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate

    result = cv2.warpPerspective(img2, Ht@H, (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1], t[0]:w1+t[0]] = img1
    return result

# ========== run on all frames
video_frames = get_frames(video_dir)
out_frames = []
for i,frame in enumerate(video_frames):
    kp_f, desc_f = feature_extractor.detectAndCompute(frame, None)
    matches = bfMatch.knnMatch(desc_f,desc_t,k = 2)
    good_and_second_good_match_list = []
    for m in matches:
        if m[0].distance/m[1].distance < 0.5:
            good_and_second_good_match_list.append(m)
    good_match_arr = np.asarray(good_and_second_good_match_list)[:,0]
    im_matches = cv2.drawMatchesKnn(frame, kp_f, template, kp_t,
                                good_and_second_good_match_list[0:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    good_kp_f = np.array([kp_f[m.queryIdx].pt for m in good_match_arr])
    good_kp_t = np.array([kp_t[m.trainIdx].pt for m in good_match_arr])
    H, masked = cv2.findHomography(good_kp_t, good_kp_f, cv2.RANSAC, 5.0)
    rgb_r_warped = warpTwoImages(frame,template,H)
    out_frames.append(rgb_r_warped)
    size_x,size_y,_ =rgb_r_warped.shape 
    # === plot 
    # ext = (0,size_y,0,size_x)
    # ax_data.set_extent(ext)
    # ax_data.set_data(rgb_r_warped)
    # fig.canvas.draw()
    # fig.canvas.flush_events()
    # plt.figure(figsize=(10, 10))
    # plt.imshow(im_matches)
    # plt.show()

# === create a video from output frames
# frames_shape = np.array([frame.shape for frame in out_frames])

# width,height = max(frames_shape[:,1]),max(frames_shape[:,0])
# fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
# video = cv2.VideoWriter('warp_out.avi', 0, 10, (int(width), int(height))) 
# for frame in out_frames:
#     frame = cv2.resize(frame,(width,height))
#     video.write(frame)

cv2.destroyAllWindows() 
video.release()