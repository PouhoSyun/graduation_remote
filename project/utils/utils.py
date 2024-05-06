# data fetching and preprocessing module

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import albumentations as alb
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from project.utils.unpack import unpack
# from unpack import unpack

TIME_PERIOD = 0.02

# 4 basic methods for numpy.array -- fundamental stack/split & fine stack/split functions
# hvstack -- array[n*n, c, size, size] -> array[c, n*size, n*size]
def hvstack(src: np.ndarray, dim4=True):
    if not dim4: 
        for i in src: i = np.array([i])
    dst = []
    for c in src.transpose(1, 0, 2, 3):
        size = int(src.shape[2])
        pieces = int(src.shape[0] ** 0.5)
        lines = []
        for line in src.reshape(pieces, pieces, size, size):
            lines.append(np.hstack(line))
        dst.append(np.vstack(lines).astype(np.uint8))
    if not dim4:
        dst = dst[0]
    return np.array(dst)

# hvsplit -- array[c, n*size, n*size] -> array[n*n, c, size, size]
def hvsplit(src: np.ndarray, pieces = 4, dim4=True):
    if not dim4:
        src = np.array([c])
    dst = []
    for c in src:
        lines = np.vsplit(c, pieces)
        dst_c = []
        for line in lines:
            line = np.hsplit(line, pieces)
            dst_c.extend(line)
        dst.append(dst_c)
    dst = np.array(dst).transpose(1, 0, 2, 3)
    if not dim4:
        for i in dst: i = i[0]
    return dst

# fine stack -- array[n*n, c, size, size] -> array[c, n*size, n*size]
def fine_stack(src: np.ndarray):
    dst = []
    for c in src.transpose(1, 0, 2, 3):
        size = int(c.shape[1])
        pieces = int(c.shape[0] ** 0.5)
        c = hvstack(c, dim4=False)
        dst_c = []
        for i in range(size * size):
            dst_c.append(c[(i//size)::size, (i%size)::size].reshape(pieces, pieces))
        dst.append(dst_c)
    dst = np.array(dst)
    return hvstack(dst)

# fine split -- array[c, n*size, n*size] -> array[n*n, c, size, size]
def fine_split(src: np.ndarray, pieces = 4):
    dst = []
    for c in src:
        size = int(c.shape[0] / pieces)
        dst_c = []
        for i in range(pieces * pieces):
            dst_c.append(c[(i//pieces)::pieces, (i%pieces)::pieces].reshape(size, size))
        dst.append(dst_c)
    dst = np.array(dst).transpose(1, 0, 2, 3)
    return dst

# make src(c*h*w) image to square shape with size=to_size by central-padding
def squarify(src: np.ndarray, to_size):
    dst = []
    for c in src:
        shape = c.shape
        up = (to_size - shape[0]) // 2
        down = to_size - shape[0] - up
        left = (to_size - shape[1]) // 2
        right = to_size - shape[1] - left
        dst.append(cv2.copyMakeBorder(c, up, down, left, right, cv2.BORDER_CONSTANT, value=(128)).tolist())
    return np.array(dst)
    
# size of event camera is 190*180
# transform matfile to stream array, then use spatial-temporal voxel grid method to record events.
# the output is the event voxel grid map resized to size_scale(default:8) times
def pack_event_stream(ev_stream, split=True, 
                      size=(260, 346)):
    if False:
        print("Reading event countmap from npy")
        event_countmap = np.load("dataset_davis/event_countmap.npy")
    else:
        event_countmap = []
        # kernel = np.array([[0.22, 0.35, 0.49, 0.35, 0.22],
        #                    [0.35, 0.49, 0.70, 0.49, 0.35],
        #                    [0.49, 0.70, 1.00, 0.70, 0.49],
        #                    [0.35, 0.49, 0.70, 0.49, 0.35],
        #                    [0.22, 0.35, 0.49, 0.35, 0.22]])
        kernel = np.array([[0.66, 0.73, 0.81, 0.73, 0.66],
                           [0.73, 0.81, 0.90, 0.81, 0.73],
                           [0.81, 0.90, 1.00, 0.90, 0.81],
                           [0.73, 0.81, 0.90, 0.81, 0.73],
                           [0.66, 0.73, 0.81, 0.73, 0.66]])
        
        with tqdm(range(len(ev_stream))) as pbar:
            for events in ev_stream:
                # DAVIS infrared senser use linear threshold
                event_field = np.zeros(size)
                for event in events:
                    #consider pre/past event affect
                    eff = -kernel * event[3] / (event[2] / TIME_PERIOD - 1)
                    x = int(event[0])
                    y = int(event[1])
                    eff = eff[max(0,2-y):min(5,size[0]+2-y),max(0,2-x):min(5,size[1]+2-x)]
                    event_field[max(0,y-2):min(size[0],y+3), max(0,x-2):min(size[1],x+3)] += eff
                std = StandardScaler()
                ef = std.fit_transform(event_field.flatten().reshape(-1, 1))
                event_field = ef.reshape(event_field.shape) * 10
                event_field = np.flip(np.flip(event_field, axis=0), axis=1)
                event_field = np.uint8(255 / (1 + np.exp(-event_field)))

                # split the output into 16 50*50 pieces, if necessary
                # event_field = cv2.resize(event_field, dsize=dsize, interpolation=cv2.INTER_LINEAR)

                event_field = cv2.dilate(event_field, np.ones((3, 3)), 5)
                event_field = cv2.erode(event_field, np.ones((7, 7)), 8)
                thr = np.sort(event_field.flatten())[-500]
                event_field = cv2.threshold(event_field, thr, 255, cv2.THRESH_TOZERO)[1]
                
                cv2.imwrite("results/sam.jpg", event_field)
                # event_field = cv2.dilate(event_field, np.ones((7, 7)), 5)
            
                event_field = np.array([event_field])
                event_field = squarify(event_field, 400)
                
                if not split:
                    event_countmap.append(event_field)
                else:
                    event_countmap.append(hvsplit(event_field))

                pbar.set_postfix()
                pbar.update(0)
        
        event_countmap = np.array(event_countmap)
        print("Building event countmap from packages")
        np.save("dataset_davis/event_countmap.npy", event_countmap)
    return event_countmap

# size of frame camera is 1520 * 1440, then split to 16 1-channel 400*400 cell-pics.
# def load_frame_png(dataset, file_id, cmap, split=True):
#     frame = cv2.imread("dataset/" + dataset + "/RGB_frame/frame" + str(file_id+1) + ".png", cmap)
#     frame = np.array([frame])
#     frame = squarify(frame, 1600)
#     if not split: return frame
#     else: return hvsplit(frame).astype(np.uint8)

# dataset_format={'raw' for pngs and matlike events, 'aedat' for davis24 datasets}
def pack_frame_png(frames_raw, split=True, size=400):
    frames = []
    for frame in frames_raw:
        frame = np.flip(frame, axis=1)
        frame = squarify(frame, size).tolist()
        if split:
            frame = hvsplit(frame)
        frames.append(frame)
    print("Frame PNGs have been packed to dump_ndarray")
    return np.array(frames)

# get frame dataset: cmap--cv2.IMREAD_*, size--square edge length of the image
class Frame_Dataset(data.Dataset):
    def __init__(self, frames_raw, size, split):
        self.images = pack_frame_png(frames_raw, split, size)
        self._length = len(self.images)
        self.split = split
        
        self.rescaler = alb.SmallestMaxSize(max_size=size)
        self.cropper = alb.CenterCrop(height=size, width=size)
        self.preprocessor = alb.Compose([self.rescaler, self.cropper])
        pass

    def __len__(self):
        if self.split: return self._length * 16
        else: return self._length
    
    def __getitem__(self, index):
        # index = index - 1
        if not self.split:
            item = self.images[index]
            item = self.preprocessor(image=item.transpose(1, 2, 0))["image"]
        else:
            item = self.images[index//16]
            item = self.preprocessor(image=item[index%16].transpose(1, 2, 0))["image"]
        item = ((item + 160) / 512 - 1).astype(np.float32)
        return torch.Tensor(item.transpose(2, 0, 1))

class Event_Dataset(data.Dataset):
    def __init__(self, ev_stream, size, split):
        self.events = pack_event_stream(ev_stream, split, size=(260, 346))
        self._length = len(self.events)
        self.split = split

        self.rescaler = alb.SmallestMaxSize(max_size=size)
        self.cropper = alb.CenterCrop(height=size, width=size)
        self.preprocessor = alb.Compose([self.rescaler, self.cropper])
        pass

    def __len__(self):
        if self.split: return self._length * 16
        else: return self._length
    
    def __getitem__(self, index):
        # index = index - 1
        if not self.split:
            item = self.events[index]
            item = self.preprocessor(image=item.transpose(1, 2, 0))["image"]
        else:
            item = self.events[index//16]
            item = self.preprocessor(image=item[index%16].transpose(1, 2, 0))["image"]
        item = (item / 127.5 - 1.0).astype(np.float32)
        item = torch.Tensor(item.transpose(2, 0, 1))
        # item = nn.functional.conv2d(item, torch.ones(1, 1, 5, 5), stride=1, padding=2)
        # item = torch.atan(item)
        return item

#dataset class for transformer, return items of 2-channels, with the shape of size*size
class DAVIS_Dataset(data.Dataset):
    def __init__(self, dataset, size, split, sam=False):
        ev_stream, frames_raw = unpack(dataset)
        if sam:
            ev_stream = ev_stream[:-1]
            frames_raw = frames_raw[1:]
        self.eventset = Event_Dataset(ev_stream, size, split)
        self.frameset = Frame_Dataset(frames_raw, size, split)
        self._length = len(self.frameset)
        self.scaler = 20
        self.sam = sam
        if sam: self.scaler = 1
        pass

    def __len__(self):
        return self._length * self.scaler
    
    def __getitem__(self, index):
        index = index % self._length
        item = np.array([self.frameset[index][0], self.eventset[index][0]])
        if self.sam:
            item = item[:, 70:330, 27:373]
        return torch.Tensor(item)
    
    def __show__(self, index):
        frame1 = np.array(self.frameset[index][0].add(1).mul(127.5))
        event1 = np.array(self.eventset[index][0].add(1).mul(127.5))
        frame2 = np.array(self.frameset[index+1][0].add(1).mul(127.5))
        event2 = np.array(self.eventset[index+1][0].add(1).mul(127.5))
        show_view = np.hstack([frame1, event1, frame2, event2])
        cv2.imwrite("results/events.jpg", show_view)

def load_frameset(args):
    _, frames_raw = unpack(args.dataset)
    dataset = Frame_Dataset(frames_raw, args.image_size, args.split)
    train_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader

def load_davisset(args):
    dataset = DAVIS_Dataset(args.dataset, args.image_size, args.split, args.sam)
    train_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    return train_loader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def plot_images(images):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()

# preview DAVIS dataset
if __name__ == '__main__':
    dataset = DAVIS_Dataset("1", 400, False)
    while(True):
        dataset.__show__(int(input("Index to preview: ")))