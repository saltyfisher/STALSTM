import os
import pandas as pd

class read_skeleton_file:
    def __init__(self):
        self.skeletonfile_path = 'D:/NTU_RGB/nturgbd_skeletons/nturgb+d_skeletons/'
        self.truthlabel=[]
        self.skeleton_data=[]

        file_list = os.listdir(self.skeletonfile_path)
        skeleton_data = {}
        for i, file_name in enumerate(file_list):
            f_name = os.path.splitext(file_name)[0]
            self.truthlabel.append(int(f_name.split('A')[1]))
            file = pd.read_table(self.skeletonfile_path + file_name, header=None)

            skeleton_data['frame_count'] = int(file[0][0])
            skeleton_data['body_info'] = []
            body_info = {}
            for f in range(skeleton_data['frame_count']):
                body_count = int(file[0][1+f*])
                for j in range(skeleton_data['body_count']):
                    info_text = file[0][2+j*27].split(' ')
                    body_info['bodyID'] = info_text[0]
                    body_info['clippedEdges'] = info_text[1]
                    body_info['handLeftConfidence'] = info_text[2]
                    body_info['handLeftState'] = info_text[3]
                    body_info['handRightConfidence'] = info_text[4]
                    body_info['handRightState'] = info_text[5]
                    body_info['isRestricted'] = info_text[6]
                    body_info['leanX'] = info_text[7]
                    body_info['leanY'] = info_text[8]
                    body_info['trackingState'] = info_text[9]
                    body_info['jointCount'] = int(file[0][3])
                    joints = {}
                    for t in range(body_info['jointCount']):
                        joint = {}
                        info_text = file[0][4+t].split(' ')
                        joint['x'] = float(info_text[0])
                        joint['y'] = float(info_text[1])
                        joint['z'] = float(info_text[2])

                        joint['depthX'] = float(info_text[3])
                        joint['depthY'] = float(info_text[4])

                        joint['colorX'] = float(info_text[5])
                        joint['colorY'] = float(info_text[6])

                        joint['orientationW'] = float(info_text[7])
                        joint['orientationX'] = float(info_text[8])
                        joint['orientationY'] = float(info_text[9])
                        joint['orientationZ'] = float(info_text[10])

                        joint['trackingState'] = info_text[11]

                        joints[t] = joint

                    body_info['joints'] = joints