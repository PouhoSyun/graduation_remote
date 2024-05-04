import scipy
import numpy as np

def array_padding(arr, to_size=(260, 346)):
    shape = (to_size[0] - arr.shape[0], to_size[1] - arr.shape[1])
    up = shape[0] // 2
    left = shape[1] // 2
    np.pad(arr, ((up, shape[0] - up), (left, shape[1] - left)),
           'constant', constant_values=(128, 128))
    return arr

def unpack(dataset_mat):
    path = "dataset_davis/"
    # data = scipy.io.loadmat(path + dataset_mat + ".mat")
    data = np.load(path + dataset_mat + '.npy', allow_pickle=True).item()
    x = data['events']['x']
    y = data['events']['y']
    t = data['events']['t'] * 1.0e-6
    p = data['events']['p'] * 2 - 1
    frames = data['frames']['s']
    frames = [np.array([array_padding(frame)]) for frame in frames]

    events = [[]]
    flag = 1
    frame_times = data['frames']['t'] * 1.0e-6
    for event in tuple(zip(x, y, t, p)):
        if(event[2] <= frame_times[flag]):
            event = (event[0], event[1], event[2] - frame_times[flag-1], event[3])
            events[-1].append(event)
        else:
            events[-1] = np.array(events[-1])
            event = (event[0], event[1], event[2] - frame_times[flag], event[3])
            events.append([event])
            flag += 1
            if(flag == len(frame_times)): break
    events[-1] = np.array(events[-1])
    print(f"Unpacking all found data of {len(events)} event-packages and {len(frames)} frames")
    return events, frames
    
if __name__ == '__main__':
    events, frames = unpack("1")
    i = []
    j = []
    for frame in frames:
        i.append(frame.max())
        j.append(frame.min())
    print(max(i), min(j))
    pass