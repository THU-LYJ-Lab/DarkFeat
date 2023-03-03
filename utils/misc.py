from pathlib import Path
import time
from collections import OrderedDict
import numpy as np
import cv2
import rawpy
import torch
import colour_demosaicing


class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default'):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1./total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()


class VideoStreamer:
    def __init__(self, basedir, resize, image_glob):
        self.listing = []
        self.resize = resize
        self.i = 0
        if Path(basedir).is_dir():
            print('==> Processing image directory input: {}'.format(basedir))
            self.listing = list(Path(basedir).glob(image_glob[0]))
            for j in range(1, len(image_glob)):
                image_path = list(Path(basedir).glob(image_glob[j]))
                self.listing = self.listing + image_path
            self.listing.sort()
            if len(self.listing) == 0:
                raise IOError('No images found (maybe bad \'image_glob\' ?)')
            self.max_length = len(self.listing)
        else:
            raise ValueError('VideoStreamer input \"{}\" not recognized.'.format(basedir))

    def load_image(self, impath):
        raw = rawpy.imread(str(impath)).raw_image_visible
        raw = np.clip(raw.astype('float32') - 512, 0, 65535)
        img = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(raw, 'RGGB').astype('float32')
        img = np.clip(img, 0, 16383)

        m = img.mean()
        d = np.abs(img - img.mean()).mean()
        img = (img - m + 2*d) / 4/d * 255
        image = np.clip(img, 0, 255)

        w_new, h_new = self.resize[0], self.resize[1]

        im = cv2.resize(image.astype('float32'), (w_new, h_new), interpolation=cv2.INTER_AREA)
        return im

    def next_frame(self):
        if self.i == self.max_length:
            return (None, False)
        image_file = str(self.listing[self.i])
        image = self.load_image(image_file)
        self.i = self.i + 1
        return (image, True)


def frame2tensor(frame, device):
    if len(frame.shape) == 2:
        return torch.from_numpy(frame/255.).float()[None, None].to(device)
    else:
        return torch.from_numpy(frame/255.).float().permute(2, 0, 1)[None].to(device)


def make_matching_plot_fast(image0, image1, mkpts0, mkpts1,
                            color, text, path=None, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    H0, W0 = image0.shape[:2]
    H1, W1 = image1.shape[:2]
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[:H1, W0+margin:, :] = image1

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    out_backup = out.copy()

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)
        
    return out / 2 + out_backup / 2

