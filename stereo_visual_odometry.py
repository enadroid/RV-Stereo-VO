import numpy as np
import cv2
from scipy.optimize import least_squares
from helpers import Euler_ZXZ_Matrix, minimize_Reprojection, generate_3D_Points
import copy
from ground_truth_data import data  #radi preglednosti ucitavamo podatke iz posebnog file-a

from plot_trajectory import plot

if __name__ == "__main__":

    startFrame = "000115"
    endFrame =  "000900"

    useSIFT = True
    
    Proj1 = np.array([[800.0, 0.0, 800.0, -800.0*0.5],[0.0, 800.0, 600.0, 0.0],[0, 0, 1, 0]])
    Proj2 = copy.deepcopy(Proj1)
    Proj1[0][3] = 0


    leftImagePath = './_out/episode_0000/Camera_left/'
    rightImagePath = './_out/episode_0000/Camera_right/'

    translation = None
    rotation = None


    calc_coordinates = []
 
    H = 1200
    W = 1200
    
    for frm in range(int(startFrame)+1, int(endFrame)+1):
        print("Frame: ", frm)


        imgPath = leftImagePath +  str(frm-1).zfill(6)+'.png'   
        ImT1_L = cv2.imread(imgPath, 0) 
        imgPath = rightImagePath + str(frm-1).zfill(6)+'.png'        
        ImT1_R = cv2.imread(imgPath, 0) 

        imgPath = leftImagePath + str(frm).zfill(6)+'.png'
        ImT2_L = cv2.imread(imgPath, 0)
        imgPath = rightImagePath + str(frm).zfill(6)+'.png'
        ImT2_R = cv2.imread(imgPath, 0)

        block = 11

        P1 = block * block * 8
        P2 = block * block * 32


        disparityEngine = cv2.StereoSGBM_create(minDisparity=0,numDisparities=32, blockSize=block, P1=P1, P2=P2)
        ImT1_disparity = disparityEngine.compute(ImT1_L, ImT1_R).astype(np.float32)
        ImT1_disparityA = np.divide(ImT1_disparity, 16.0)

        ImT2_disparity = disparityEngine.compute(ImT2_L, ImT2_R).astype(np.float32)
        ImT2_disparityA = np.divide(ImT2_disparity, 16.0)

        TILE_H = 10
        TILE_W = 20

        if useSIFT:
            featureEngine = cv2.xfeatures2d.SIFT_create()
           
        else:
            featureEngine = cv2.FastFeatureDetector_create()
            

        H,W = ImT1_L.shape
        if useSIFT:
            kp = featureEngine.detect(ImT1_L)
        else:

            kp = []
            idx = 0
            for y in range(0, H, TILE_H):
                for x in range(0, W, TILE_W):
                    imPatch = ImT1_L[y:y+TILE_H, x:x+TILE_W]
                    keypoints = featureEngine.detect(imPatch)
                    for pt in keypoints:
                        pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

                    if (len(keypoints) > 10):
                        keypoints = sorted(keypoints, key=lambda x: -x.response)
                        for kpt in keypoints[0:10]:
                            kp.append(kpt)
                    else:
                        for kpt in keypoints:
                            kp.append(kpt)
      
        

        trackPoints1 = cv2.KeyPoint_convert(kp)
        trackPoints1 = np.expand_dims(trackPoints1, axis=1)

        # Parametri za lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                          maxLevel = 3,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

        trackPoints2, st, err = cv2.calcOpticalFlowPyrLK(ImT1_L, ImT2_L, trackPoints1, None, flags=cv2.MOTION_AFFINE, **lk_params)

        ptTrackable = np.where(st == 1, 1,0).astype(bool)
        trackPoints1_KLT = trackPoints1[ptTrackable, ...]
        trackPoints2_KLT_t = trackPoints2[ptTrackable, ...]
        trackPoints2_KLT = np.around(trackPoints2_KLT_t)

        error = 4
        errTrackablePoints = err[ptTrackable, ...]
        errThresholdedPoints = np.where(errTrackablePoints < error, 1, 0).astype(bool)
        trackPoints1_KLT = trackPoints1_KLT[errThresholdedPoints, ...]
        trackPoints2_KLT = trackPoints2_KLT[errThresholdedPoints, ...]

  
        hPts = np.where(trackPoints2_KLT[:,1] >= H)
        wPts = np.where(trackPoints2_KLT[:,0] >= W)
        outTrackPts = hPts[0].tolist() + wPts[0].tolist()
        outDeletePts = list(set(outTrackPts))

        if len(outDeletePts) > 0:
            trackPoints1_KLT_L = np.delete(trackPoints1_KLT, outDeletePts, axis=0)
            trackPoints2_KLT_L = np.delete(trackPoints2_KLT, outDeletePts, axis=0)
        else:
            trackPoints1_KLT_L = trackPoints1_KLT
            trackPoints2_KLT_L = trackPoints2_KLT

       
        pointDiff = trackPoints1_KLT_L - trackPoints2_KLT_L
        pointDiffSum = np.sum(np.linalg.norm(pointDiff))

        trackPoints1_KLT_R = np.copy(trackPoints1_KLT_L)
        trackPoints2_KLT_R = np.copy(trackPoints2_KLT_L)
        selectedPointMap = np.zeros(trackPoints1_KLT_L.shape[0])

        disparityMinThres = 0.0
        disparityMaxThres = 100.0

        for i in range(trackPoints1_KLT_L.shape[0]):
            T1Disparity = ImT1_disparityA[int(trackPoints1_KLT_L[i,1]), int(trackPoints1_KLT_L[i,0])]
            T2Disparity = ImT2_disparityA[int(trackPoints2_KLT_L[i,1]), int(trackPoints2_KLT_L[i,0])]

            if (T1Disparity > disparityMinThres and T1Disparity < disparityMaxThres
                and T2Disparity > disparityMinThres and T2Disparity < disparityMaxThres):
                trackPoints1_KLT_R[i, 0] = trackPoints1_KLT_L[i, 0] - T1Disparity
                trackPoints2_KLT_R[i, 0] = trackPoints2_KLT_L[i, 0] - T2Disparity
                selectedPointMap[i] = 1

        selectedPointMap = selectedPointMap.astype(bool)
        trackPoints1_KLT_L_3d = trackPoints1_KLT_L[selectedPointMap, ...]
        trackPoints1_KLT_R_3d = trackPoints1_KLT_R[selectedPointMap, ...]
        trackPoints2_KLT_L_3d = trackPoints2_KLT_L[selectedPointMap, ...]
        trackPoints2_KLT_R_3d = trackPoints2_KLT_R[selectedPointMap, ...]

        # 3d point cloud triagulacija
        numPoints = trackPoints1_KLT_L_3d.shape[0]
        d3dPointsT1 = generate_3D_Points(trackPoints1_KLT_L_3d, trackPoints1_KLT_R_3d, Proj1, Proj2)
        d3dPointsT2 = generate_3D_Points(trackPoints2_KLT_L_3d, trackPoints2_KLT_R_3d, Proj1, Proj2)
        
        ransacError = float('inf')
        dOut = None
        # RANSAC
        ransacSize = 6
        for ransacItr in range(250):
            sampledPoints = np.random.randint(0, numPoints, ransacSize)
            rD2dPoints1_L = trackPoints1_KLT_L_3d[sampledPoints]
            rD2dPoints2_L = trackPoints2_KLT_L_3d[sampledPoints]
            rD3dPointsT1 = d3dPointsT1[sampledPoints]
            rD3dPointsT2 = d3dPointsT2[sampledPoints]

            dSeed = np.zeros(6)
 
            optRes = least_squares(minimize_Reprojection, dSeed, method='lm', max_nfev=200,
                                args=(rD2dPoints1_L, rD2dPoints2_L, rD3dPointsT1, rD3dPointsT2, Proj1))

  
            error = minimize_Reprojection(optRes.x, trackPoints1_KLT_L_3d, trackPoints2_KLT_L_3d,
                                            d3dPointsT1, d3dPointsT2, Proj1)

            eCoords = error.reshape((d3dPointsT1.shape[0]*2, 3))
            totalError = np.sum(np.linalg.norm(eCoords, axis=1))

            if (totalError < ransacError):
                ransacError = totalError
                dOut = optRes.x


        Rmat = Euler_ZXZ_Matrix(dOut[0], dOut[1], dOut[2])
        translationArray = np.array([[dOut[3]], [dOut[4]], [dOut[5]]])
        

                   
        if (isinstance(translation, np.ndarray)):
            translation = translation + np.matmul(rotation, translationArray)
        else:
            translation = translationArray

        if (isinstance(rotation, np.ndarray)):
            rotation = np.matmul(Rmat, rotation)
        else:
            rotation = Rmat



        WCorr = 290
        HCorr = 200
        x, y = int(translation[0])+WCorr, int(translation[2])+HCorr
           
        
        calc_coordinates.append([x,y])

            
        if frm == int(endFrame):
            outF = open("calculated_coordinates.txt", "w")  #spasavamo rezultate za plotanje
            outF.write(str(calc_coordinates))
            outF.close()
            ground_truth_traj = data()
            plot(calc_coordinates, ground_truth_traj)




