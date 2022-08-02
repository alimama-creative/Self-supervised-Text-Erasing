from __future__ import division
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import scipy.signal as ssig
import pygame, pygame.locals
from pygame import freetype
#import Image
import math

def sample_weighted(p_dict):
    ps = list(p_dict.keys())
    return p_dict[np.random.choice(ps,p=ps)]

def move_bb(bbs, t):
    """
    Translate the bounding-boxes in by t_x,t_y.
    BB : 2x4xn
    T  : 2-long np.array
    """
    return bbs + t[:,None,None]

def crop_safe(arr, rect, bbs=[], pad=0):
    """
    ARR : arr to crop
    RECT: (x,y,w,h) : area to crop to
    BBS : nx4 xywh format bounding-boxes
    PAD : percentage to pad

    Does safe cropping. Returns the cropped rectangle and
    the adjusted bounding-boxes
    """
    rect = np.array(rect)
    rect[:2] -= pad
    rect[2:] += 2*pad
    v0 = [max(0,rect[0]), max(0,rect[1])]
    v1 = [min(arr.shape[0], rect[0]+rect[2]), min(arr.shape[1], rect[1]+rect[3])]
    arr = arr[v0[0]:v1[0],v0[1]:v1[1],...]
    if len(bbs) > 0:
        for i in range(len(bbs)):
            bbs[i,0] -= v0[0]
            bbs[i,1] -= v0[1]
        return arr, bbs
    else:
        return arr


class BaselineState(object):
    curve = lambda this, a: lambda x: a*x*x
    differential = lambda this, a: lambda x: 2*a*x
    a = [0.50, 0.05]

    def get_sample(self):
        """
        Returns the functions for the curve and differential for a and b
        """
        sgn = 1.0
        if np.random.rand() < 0.5:
            sgn = -1

        a = self.a[1]*np.random.randn() + sgn*self.a[0]
        return {
            'curve': self.curve(a),
            'diff': self.differential(a),
        }

class RenderFont(object):
    """
    Outputs a rasterized font sample.
        Output is a binary mask matrix cropped closesly with the font.
        Also, outputs ground-truth bounding boxes and text string
    """

    def __init__(self, data_dir='data'):
        # distribution over the type of text:
        # whether to get a single word, paragraph or a line:
        # self.p_curved = 0
        self.baselinestate = BaselineState()

    def init_font(self, fs):
        """
        Initializes a pygame font.
        FS : font-state sample
        """
        font = freetype.Font(fs['font'], size=fs['size'])
        font.underline = fs['underline']
        # font.underline_adjustment = max(2.0, min(-2.0, 2.0*np.random.randn() + 1.0)),
        font.strong = fs['strong']
        font.oblique = fs['oblique']
        font.strength = (1.0 - 0.05)*np.random.rand() + 0.05
        # char_spacing = fs['char_spacing']
        font.antialiased = True
        font.origin = True
        return font

    def render_multiline(self,font,text,direction=0):
        """
        renders multiline TEXT on the pygame surface SURF with the
        font style FONT.
        A new line in text is denoted by \n, no other characters are 
        escaped. Other forms of white-spaces should be converted to space.

        returns the updated surface, words and the character bounding boxes.
        """
        # get the number of lines
        lines = text.split('\n')
        lengths = [len(l) for l in lines]

        # font parameters:
        line_spacing = font.get_sized_height() + 1
        # line_spacing = font.get_sized_width() + 1
        # initialize the surface to proper size:
        line_bounds = font.get_rect(lines[np.argmax(lengths)])
        
        if direction == 0:
            fsize = (round(2.0*line_bounds.width), round(1.25*line_spacing*len(lines)))
        else:
            fsize = (round(1.25*line_spacing*len(lines)), round(3.0*line_spacing))

        surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)
        bbs = []
        space = font.get_rect('0')
        x, y = -line_spacing, 0
        for l in lines:
            if direction == 0:
                x = 0 # carriage-return
                y += line_spacing # line-feed
            else:
                x += line_spacing
                y = line_spacing
            for ch in l: # render each character
                if ch.isspace(): # just shift
                    if direction == 0:  
                        x += space.width+1
                    else:
                        y += space.width
                else:
                    # render the character
                    ch_bounds = font.render_to(surf, (x,y), ch)

                    ch_bounds.x = x + ch_bounds.x
                    ch_bounds.y = y - ch_bounds.y
                    if direction == 0: 
                        # x += ch_bounds.width + 4
                        extern_dis = np.random.randint(0,4)
                        x += ch_bounds.width + extern_dis
                    else:
                        # y += ch_bounds.width + 3
                        extern_dis = np.random.randint(0,4)
                        y += ch_bounds.width + extern_dis
                    bbs.append(np.array(ch_bounds))


        # get the union of characters for cropping:
        r0 = pygame.Rect(bbs[0])
        rect_union = r0.unionall(bbs)
        # get the words:
        words = ' '.join(text.split())

        # crop the surface to fit the text:
        bbs = np.array(bbs)
        surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=5)
        surf_arr = surf_arr.swapaxes(0,1)
        #self.visualize_bb(surf_arr,bbs)
        return surf_arr, words, bbs

    def render_curved(self, font, word_text, fs):
        """
        use curved baseline for rendering word
        """
        wl = len(word_text)
        if wl < 5 or fs["curved"] == 0:
            # print(word_text, fs["direction"])
            return self.render_multiline(font, word_text, fs["direction"])

        # create the surface:
        lspace = font.get_sized_height() + 1
        lbound = font.get_rect(word_text)
        fsize = (round(2.0*lbound.width), round(3*lspace))
        surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

        # baseline state
        mid_idx = wl//2
        BS = self.baselinestate.get_sample()
        curve = [BS['curve'](i-mid_idx) for i in range(wl)]
        curve[mid_idx] = -np.sum(curve) / (wl-1)
        rots  = [-int(math.degrees(math.atan(BS['diff'](i-mid_idx)/(font.size/2)))) for i in range(wl)]

        bbs = []
        # place middle char
        rect = font.get_rect(word_text[mid_idx])
        rect.centerx = surf.get_rect().centerx
        rect.centery = surf.get_rect().centery + rect.height
        rect.centery +=  curve[mid_idx]
        ch_bounds = font.render_to(surf, rect, word_text[mid_idx], rotation=rots[mid_idx])
        ch_bounds.x = rect.x + ch_bounds.x
        ch_bounds.y = rect.y - ch_bounds.y
        mid_ch_bb = np.array(ch_bounds)

        # render chars to the left and right:
        last_rect = rect
        ch_idx = []
        for i in range(wl):
            #skip the middle character
            if i==mid_idx: 
                bbs.append(mid_ch_bb)
                ch_idx.append(i)
                continue

            if i < mid_idx: #left-chars
                i = mid_idx-1-i
            elif i==mid_idx+1: #right-chars begin
                last_rect = rect

            ch_idx.append(i)
            ch = word_text[i]

            newrect = font.get_rect(ch)
            newrect.y = last_rect.y
            if i > mid_idx:
                newrect.topleft = (last_rect.topright[0]+2, newrect.topleft[1])
            else:
                newrect.topright = (last_rect.topleft[0]-2, newrect.topleft[1])
            newrect.centery = max(newrect.height, min(fsize[1] - newrect.height, newrect.centery + curve[i]))
            try:
                bbrect = font.render_to(surf, newrect, ch, rotation=rots[i])
            except ValueError:
                bbrect = font.render_to(surf, newrect, ch)
            except TypeError:    
                return self.render_multiline(font, word_text, fs["direction"])
            bbrect.x = newrect.x + bbrect.x
            bbrect.y = newrect.y - bbrect.y
            bbs.append(np.array(bbrect))
            last_rect = newrect
        
        # correct the bounding-box order:
        bbs_sequence_order = [None for i in ch_idx]
        for idx,i in enumerate(ch_idx):
            bbs_sequence_order[i] = bbs[idx]
        bbs = bbs_sequence_order

        # get the union of characters for cropping:
        r0 = pygame.Rect(bbs[0])
        rect_union = r0.unionall(bbs)

        # crop the surface to fit the text:
        bbs = np.array(bbs)
        # change the surface to text mask
        surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=5)
        surf_arr = surf_arr.swapaxes(0,1)
        return surf_arr, word_text, bbs

    def place_text(self, text_arrs, back_arr, pos, bbs):
        areas = [-np.prod(ta.shape) for ta in text_arrs]
        order = np.argsort(areas)

        locs = [None for i in range(len(text_arrs))]
        out_arr = np.zeros_like(back_arr)
        for i in order:        
            loc = pos
            # update the bounding-boxes:
            # bbs[i] = move_bb(bbs[i],loc[::-1])

            # blit the text onto the canvas
            w,h = text_arrs[i].shape
            w, h = min(w, out_arr.shape[0]-loc[1]), min(h, out_arr.shape[1]-loc[0])
            out_arr[loc[1]:loc[1]+w,loc[0]:loc[0]+h] += text_arrs[i][:w,:h]

        return out_arr, locs, bbs, order

    def bb_xywh2coords(self,bbs):
        """
        Takes an nx4 bounding-box matrix specified in x,y,w,h
        format and outputs a 2x4xn bb-matrix, (4 vertices per bb).
        """
        n,_ = bbs.shape
        coords = np.zeros((2,4,n))
        for i in range(n):
            coords[:,:,i] = bbs[i,:2][:,None]
            coords[0,1,i] += bbs[i,2]
            coords[:,2,i] += bbs[i,2:4]
            coords[1,3,i] += bbs[i,3]
        return coords

    def render_sample(self, word, font, mask, pos, fs):
        """
        Places text in the "collision-free" region as indicated
        in the mask -- 255 for unsafe, 0 for safe.
        The text is rendered using FONT, the text content is TEXT.
        """
        font.size = fs["size"] # set the font-size
        # sample text:
        text = word
        # render the text:
        txt_arr, txt, bb = self.render_curved(font, text, fs)
        if len(text)>3 and fs["warp_perspective"]:
            # self.visualize_bb(txt_arr, bb)
            # cv2.imwrite("pic/bbs1.png", txt_arr)
            txt_arr = warpGraphy(txt_arr)
            # cv2.imwrite("pic/bbs2.png", txt_arr)
            # print("save !!!!!!!!!!!!!!!!!!!")
        bb = self.bb_xywh2coords(bb)
        # position the text within the mask:
        text_mask, loc, bb, _ = self.place_text([txt_arr], mask, pos, [bb])
        return text_mask, bb

    def visualize_bb(self, text_arr, bbs):
        ta = text_arr.copy()
        for r in bbs:
            cv2.rectangle(ta, (int(r[0]),int(r[1])), (int(r[0]+r[2]),int(r[1]+r[3])), color=128, thickness=1)
        cv2.imwrite("pic/bbs.png", ta)

def warpGraphy(src_mat):    
    # print(src_mat.shape)
    h, w = src_mat.shape[0:2]
    pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    if np.random.rand()>0.75:
        pts2 = np.float32([[int(np.random.rand()/4*w),int(np.random.rand()/4*h)],[int((3/4+np.random.rand()/4)*w),int(np.random.rand()/4*h)],\
            [int(np.random.rand()/4*w),int((3/4+np.random.rand()/4)*h)],[int((3/4+np.random.rand()/4)*w),int((3/4+np.random.rand()/4)*h)]])
    else:
        h = int(2.5*h)
        len_l, len_r = np.random.rand()/1.66+0.4, np.random.rand()/1.66+0.4
        begin_l, begin_r = (1-len_l)*np.random.rand(), (1-len_r)*np.random.rand()
        pts2 = np.float32([[0,int(begin_l*h)],[w,int(begin_r*h)],[0,int((begin_l+len_l)*h)],[w,int((begin_r+len_r)*h)]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst_mat = cv2.warpPerspective(src_mat, M, (w,h),
                                    flags=cv2.INTER_LINEAR)
    # cv2.rectangle(dst_mat,(1,1),(w,h),128,thickness=2)
    return dst_mat