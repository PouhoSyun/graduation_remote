import scipy
import numpy as np

path = 'dataset_davis/3'
data = scipy.io.loadmat(path+'.mat')['aedat']['data'][0][0]
polarity = data['polarity'][0][0]
x = polarity['x'][0][0]
x = x.reshape(x.size)
y = polarity['y'][0][0]
y = y.reshape(y.size)
t = polarity['timeStamp'][0][0]
t = t.reshape(t.size)
p = polarity['polarity'][0][0]
p = p.reshape(p.size)
polarity = {'x':x, 'y':y, 't':t, 'p':p}
print(f"Got {len(x)} events, with the first{x[0]}-{y[0]}-{t[0]}-{p[0]}")

frame = data['frame'][0][0]
s = frame['samples'][0][0]
ts = frame['timeStampStart'][0][0]
te = frame['timeStampEnd'][0][0]
frames = []
for f in s:
    frames.append(f[0])
ts = ts.reshape(ts.size)
te = te.reshape(te.size)
t = (ts+te)//2
fra = {'s':frames, 't':t}
print(f"Got {len(frames)} frames")

data = {'events':polarity, 'frames':fra}

np.save(path+'.npy', data)
pass
