import numpy as np
import os
import imageio

def list_pics(path, pic_cnt=100, size=False):
    filenames = os.listdir(path)[:pic_cnt]
    # 按照epoch_no:batch_no升序排列
    # filenames.sort(key=lambda x:int((x.split('_')[0]))*114514+int((x.split('_')[1]).split('.')[0]))
    frames = []
    for image_name in filenames:
        im = imageio.imread(path + '/' + image_name)
        if size: im = np.resize(im, size)
        frames.append(im)
    return frames

def make_gif(frames, dump_path, duration=0.05):
    print(f"Get {len(frames)} pics, packing GIF...")
    imageio.mimsave(dump_path, frames, 'GIF', duration=duration)
    print("Packed GIF has been saved to dump_path")

if __name__ == '__main__':
    frames = []
    for i in range(50):
        frames.extend(list_pics("results/mixer/1/"+str(i)))
    make_gif(frames[::3],
             "results/result_gifs/test.gif")