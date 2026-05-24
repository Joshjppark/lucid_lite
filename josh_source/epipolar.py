
import numpy as np
import matplotlib.pyplot as plt

from PySide6.QtGui import QImage


from .geometry import *
# import geometry

def qimg_to_np(qimg: QImage):
    img = qimg.convertToFormat(QImage.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8, count=img.sizeInBytes())
    arr = arr.reshape(h, img.bytesPerLine())[:, : w * 3]                         
    return arr.reshape(h, w, 3).copy() 


def display_cams(window, cam_names, df, SKEL, TRACK_COLOR_MAP, grid_shape=None, *args):

    """
    sample func call
    display_cams(window, ['back', 'backL', 'top', 'topL'], df, (2, 2))
    """

    if grid_shape is None:
        grid_shape = (1, len(cam_names))
    
    fig, axes = plt.subplots(*grid_shape, figsize=(14, 12))
    fig.subplots_adjust(wspace=0.01, hspace=0.01)

    axes = np.array(axes).ravel()
    for cam, ax in zip(cam_names, axes):

        # get frame video data
        qimg = window._video_panels[cam]._decoder.get_frame(window._current_frame)
        frame = qimg_to_np(qimg)

        # display frame
        ax.imshow(frame)
        ax.set_title(cam)
        ax.axis('off')

        # display instances
        cam_df = df.query(f"cam == @cam")
        for pts, track in zip(cam_df['points'], cam_df['track']):
        
            # get track color of the instance
            track_color = TRACK_COLOR_MAP[track]
        

            # edges
            for edge in SKEL.edges:
                p0 = pts[edge[0]]
                p1 = pts[edge[1]]
                if p0 is None or p1 is None: continue

                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=1, color=track_color)

            # nodes
            xs = [p[0] for p in pts if p is not None]
            ys = [p[1] for p in pts if p is not None]
            ax.scatter(xs, ys, s=2, color=track_color)



def draw_epline_single_point(window, fg ,CAMERAS, cam_names, df, node_idx, track_idx=0):

    '''
    Draws a 2x2 grid, where each row has the two cameras and
    a point is drawn on both views and projected epipolar lines are displayed

    sample use:
    draw_epipolar_lines(['back', 'backL'], df, 0)
    '''

    cam1 = CAMERAS[cam_names[0]] 
    cam2 = CAMERAS[cam_names[1]]

    F1 = calc_fundamental_matrix(cam1, cam2)
    F2 = calc_fundamental_matrix(cam2, cam1)


    # pts1 = df.query('cam == @cam_names[0] and track_idx == @track_idx')['points'].iloc[0]
    # pts2 = df.query('cam == @cam_names[1] and track_idx == @track_idx')['points'].iloc[0]

    pts1 = df.query('cam == @cam_names[0] and track_idx == 3')['points'].iloc[0]
    pts2 = df.query('cam == @cam_names[1] and track_idx == 2')['points'].iloc[0]

    instance_1_pts = np.array([p if p is not None else (np.nan, np.nan) for p in pts1])
    instance_2_pts = np.array([p if p is not None else (np.nan, np.nan) for p in pts2])



    points = [instance_1_pts, instance_2_pts]


    line1 = F1.T @ homogenize(instance_2_pts[node_idx])
    line2 = F2.T @ homogenize(instance_1_pts[node_idx])

    # print(f'line1 shape: {line1.shape}')
    # print(f'line2 shape: {line2.shape}')

    lines = [line1, line2]

    err1 = homogenize(instance_1_pts[node_idx]).T @ line1 / (np.linalg.norm(line1[:2])) 
    err2 = homogenize(instance_2_pts[node_idx]).T @ line2 / (np.linalg.norm(line2[:2]))
    errors = [err1, err2]

    

    fig = plt.figure(figsize=(14, 12), layout='constrained')
    subfigs = fig.subfigures(2, 1)
    
    for fig_idx in range(2):
        subfigs[fig_idx].suptitle(f'Error from projecting onto {cam_names[fig_idx]}: {errors[fig_idx].item():.4f}', fontsize=14, fontweight='bold')
        axs = subfigs[fig_idx].subplots(1, 2)

        for cam, ax, instance_pts in zip(cam_names, axs, points):
            
            # get frame video data
            qimg = window._video_panels[cam]._decoder.get_frame(fg.frame_idx)
            frame = qimg_to_np(qimg)

            # display frame
            ax.imshow(frame)
            ax.set_title(cam)
            ax.axis('off')

            # display points
            ax.scatter(instance_pts[:, 0], instance_pts[:, 1], s=10)
            ax.scatter(instance_pts[node_idx, 0], instance_pts[node_idx, 1], s=10, c='r')

            # display the line in the second camera
            if cam_names.index(cam) == fig_idx:
                height, width = frame.shape[:2]
                a, b, c = lines[fig_idx]
                if abs(b) > abs(a):
                    x_s = [0, width]
                    y_s = [-c/b, -(a * width + c) / b]
                else:
                    x_s = [-c/a, -(b * height + c) / a]
                    y_s = [0, height]
                ax.plot(x_s, y_s, lw=2)
    
    return None


def draw_eplines_whole_instance(window, CAMERAS, cam_names, df, track_idx=0):

    '''
    Draws a 2x2 grid, where each row has the two cameras and
    a point is drawn on both views and projected epipolar lines are displayed

    sample use:
    draw_epipolar_lines(['back', 'backL'], df, 0)
    '''

    cam1 = CAMERAS[cam_names[0]] 
    cam2 = CAMERAS[cam_names[1]]

    F2 = calc_fundamental_matrix(cam1, cam2)
    # F2 = calc_fundamental_matrix(cam2, cam1)
    F1 = F2.T

    pts_1 = df.query('cam == @cam_names[0] and track_idx == 3')['points'].iloc[0]
    pts_2 = df.query('cam == @cam_names[1] and track_idx == 2')['points'].iloc[0]
    instance_1_pts = np.asarray([p if p is not None else (np.nan, np.nan) for p in pts_1])
    instance_2_pts = np.asarray([p if p is not None else (np.nan, np.nan) for p in pts_2])
    points = [instance_1_pts, instance_2_pts]

    # shape of lines: (3, n), n = 15
    lines1 = F1 @ homogenize(instance_2_pts).T
    lines2 = F2 @ homogenize(instance_1_pts).T

    # print(f'lines1 shape: {lines1.shape}')
    # print(f'lines2 shape: {lines2.shape}')

    # from josh_source.geometry import calc_epipolar_lines
    # lines1 = calc_epipolar_lines(F1, homogenize(instance_2_pts))
    # lines2 = calc_epipolar_lines(F2, homogenize(instance_1_pts))


    print(f'lines1 shape: {lines1.shape}')
    print(f'lines2 shape: {lines2.shape}')

    lines = [lines1, lines2]

    err1 = np.sum(homogenize(instance_1_pts) * lines1.T, axis=1) / (np.linalg.norm(lines1.T[:, :2], axis=1)) 
    err2 = np.sum(homogenize(instance_2_pts) * lines2.T, axis=1) / (np.linalg.norm(lines2.T[:, :2], axis=1))
    a = np.sum(homogenize(instance_1_pts) * lines1.T, axis=1)
    b = np.linalg.norm(lines1[:, :2], axis=1)
    c = (np.linalg.norm(lines2.T[:, :2], axis=1))

    errors = [err1, err2]

    colors = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", 
        "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4", 
        "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000"
        ]
    

    fig = plt.figure(figsize=(14, 12), layout='constrained')
    subfigs = fig.subfigures(2, 1)
    
    for fig_idx in range(2):
        axs = subfigs[fig_idx].subplots(1, 2)

        for cam, ax, instance_pts in zip(cam_names, axs, points):
            
            # get frame video data
            qimg = window._video_panels[cam]._decoder.get_frame(window._current_frame)
            frame = qimg_to_np(qimg)

            # display frame
            ax.imshow(frame)
            ax.set_title(cam)
            ax.axis('off')

            # display points
            ax.scatter(instance_pts[:, 0], instance_pts[:, 1], s=10, c=colors)


            # display the line in the second camera
            if cam_names.index(cam) == fig_idx:
                for node_idx in range(len(colors)):
                    height, width = frame.shape[:2]
                    line = lines[fig_idx][:, node_idx]
                    if np.nan in line: continue
                    a, b, c = line
                    if abs(b) > abs(a):
                        x_s = [0, width]
                        y_s = [-c/b, -(a * width + c) / b]
                    else:
                        x_s = [-c/a, -(b * height + c) / a]
                        y_s = [0, height]
                    ax.plot(x_s, y_s, lw=0.5, c=colors[node_idx])
    
    return errors
