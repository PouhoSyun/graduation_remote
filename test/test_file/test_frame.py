import os, sys

root_path = os.path.abspath(__file__).split('/')[:-3]
# root_path.append("project")
root_path = '/'.join(root_path)
sys.path.append(root_path)

import cv2
from project.utils.unpack import unpack
from test_gif import make_gif

def show_frames():
    _, frames = unpack("2")
    st = []
    for frame in frames[:100]:
        st.append(frame[0])
        st[-1].resize(260, 346)
    make_gif(st, "results/result_gifs/frames.gif")

if __name__ == '__main__':
    show_frames()