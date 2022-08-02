import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.interpolate as si
import scipy.ndimage as scim 
import scipy.ndimage.interpolation as sii
import os
import os.path as osp
#import cPickle as cp
import _pickle as cp
import cv2
#import Image
from PIL import Image
# from poisson_reconstruct import blit_images
import pickle
import scipy.fftpack
import scipy.ndimage

def DST(x):
    """
    Converts Scipy's DST output to Matlab's DST (scaling).
    """
    X = scipy.fftpack.dst(x,type=1,axis=0)
    return X/2.0

def IDST(X):
    """
    Inverse DST. Python -> Matlab
    """
    n = X.shape[0]
    x = np.real(scipy.fftpack.idst(X,type=1,axis=0))
    return x/(n+1.0)

def get_grads(im):
    """
    return the x and y gradients.
    """
    [H,W] = im.shape
    Dx,Dy = np.zeros((H,W),'float32'), np.zeros((H,W),'float32')
    j,k = np.atleast_2d(np.arange(0,H-1)).T, np.arange(0,W-1)
    Dx[j,k] = im[j,k+1] - im[j,k]
    Dy[j,k] = im[j+1,k] - im[j,k]
    return Dx,Dy

def get_laplacian(Dx,Dy):
    """
    return the laplacian
    """
    [H,W] = Dx.shape
    Dxx, Dyy = np.zeros((H,W)), np.zeros((H,W))
    j,k = np.atleast_2d(np.arange(0,H-1)).T, np.arange(0,W-1)
    Dxx[j,k+1] = Dx[j,k+1] - Dx[j,k] 
    Dyy[j+1,k] = Dy[j+1,k] - Dy[j,k]
    return Dxx+Dyy

def poisson_solve(gx,gy,bnd):
    # convert to double:
    gx = gx.astype('float32')
    gy = gy.astype('float32')
    bnd = bnd.astype('float32')
 
    H,W = bnd.shape
    L = get_laplacian(gx,gy)

    # set the interior of the boundary-image to 0:
    bnd[1:-1,1:-1] = 0
    # get the boundary laplacian:
    L_bp = np.zeros_like(L)
    L_bp[1:-1,1:-1] = -4*bnd[1:-1,1:-1] \
                      + bnd[1:-1,2:] + bnd[1:-1,0:-2] \
                      + bnd[2:,1:-1] + bnd[0:-2,1:-1] # delta-x
    L = L - L_bp
    L = L[1:-1,1:-1]

    # compute the 2D DST:
    L_dst = DST(DST(L).T).T #first along columns, then along rows

    # normalize:
    [xx,yy] = np.meshgrid(np.arange(1,W-1),np.arange(1,H-1))
    D = (2*np.cos(np.pi*xx/(W-1))-2) + (2*np.cos(np.pi*yy/(H-1))-2)
    L_dst = L_dst/D

    img_interior = IDST(IDST(L_dst).T).T # inverse DST for rows and columns

    img = bnd.copy()

    img[1:-1,1:-1] = img_interior

    return img

def blit_images(im_top,im_back,scale_grad=1.0,mode='max'):
    """
    combine images using poission editing.
    IM_TOP and IM_BACK should be of the same size.
    """
    assert np.all(im_top.shape==im_back.shape)

    im_top = im_top.copy().astype('float32')
    im_back = im_back.copy().astype('float32')
    im_res = np.zeros_like(im_top)

    # frac of gradients which come from source:
    for ch in range(im_top.shape[2]):
        ims = im_top[:,:,ch]
        imd = im_back[:,:,ch]

        [gxs,gys] = get_grads(ims)
        [gxd,gyd] = get_grads(imd)

        gxs *= scale_grad
        gys *= scale_grad

        gxs_idx = gxs!=0
        gys_idx = gys!=0
        # mix the source and target gradients:
        if mode=='max':
            gx = gxs.copy()
            gxm = (np.abs(gxd))>np.abs(gxs)
            gx[gxm] = gxd[gxm]

            gy = gys.copy()
            gym = np.abs(gyd)>np.abs(gys)
            gy[gym] = gyd[gym]

            # get gradient mixture statistics:
            f_gx = np.sum((gx[gxs_idx]==gxs[gxs_idx]).flat) / (np.sum(gxs_idx.flat)+1e-6)
            f_gy = np.sum((gy[gys_idx]==gys[gys_idx]).flat) / (np.sum(gys_idx.flat)+1e-6)
            if min(f_gx, f_gy) <= 0.35:
                m = 'max'
                if scale_grad > 1:
                    m = 'blend'
                return blit_images(im_top, im_back, scale_grad=1.5, mode=m)

        elif mode=='src':
            gx,gy = gxd.copy(), gyd.copy()
            gx[gxs_idx] = gxs[gxs_idx]
            gy[gys_idx] = gys[gys_idx]

        elif mode=='blend': # from recursive call:
            # just do an alpha blend
            gx = gxs+gxd
            gy = gys+gyd

        im_res[:,:,ch] = np.clip(poisson_solve(gx,gy,imd),0,255)

    return im_res.astype('uint8')


def sample_weighted(p_dict):
    ps = p_dict.keys()
    return ps[np.random.choice(len(ps),p=p_dict.values())]

class Layer(object):

    def __init__(self,alpha,color):

        # alpha for the whole image:
        # if alpha.ndim==2:
        #     print("aaa")
        assert alpha.ndim==2
        self.alpha = alpha
        [n,m] = alpha.shape[:2]

        color=np.atleast_1d(np.array(color)).astype('uint8')
        # color for the image:
        if color.ndim==1: # constant color for whole layer
            ncol = color.size
            if ncol == 1 : #grayscale layer
                self.color = color * np.ones((n,m,3),'uint8')
            if ncol == 3 : 
                self.color = np.ones((n,m,3),'uint8') * color[None,None,:]
        elif color.ndim==2: # grayscale image
            self.color = np.repeat(color[:,:,None],repeats=3,axis=2).copy().astype('uint8')
        elif color.ndim==3: #rgb image
            self.color = color.copy().astype('uint8')
        else:
            print(color.shape)
            raise Exception("color datatype not understood")


class Colorize(object):

    def __init__(self, model_dir='data'):#, im_path):
        # # get a list of background-images:
        # imlist = [osp.join(im_path,f) for f in os.listdir(im_path)]
        # self.bg_list = [p for p in imlist if osp.isfile(p)]
        # probabilities of different text-effects:
        self.p_bevel = 0.05 # add bevel effect to text
        self.p_outline = 0.05 # just keep the outline of the text
        self.p_drop_shadow = 0.15
        self.p_border = 0.15
        # self.p_drop_shadow = 0
        # self.p_border = 0
        self.p_displacement = 0.30 # add background-based bump-mapping
        self.p_texture = 0.0 # use an image for coloring text


    def drop_shadow(self, alpha, theta, shift, size, op=0.80):
        """
        alpha : alpha layer whose shadow need to be cast
        theta : [0,2pi] -- the shadow direction
        shift : shift in pixels of the shadow
        size  : size of the GaussianBlur filter
        op    : opacity of the shadow (multiplying factor)

        @return : alpha of the shadow layer
                  (it is assumed that the color is black/white)
        """
        if size == 0:
            shadow = alpha
        else:
            if size%2==0:
                size -= 1
                size = max(1,size)
            shadow = cv.GaussianBlur(alpha,(size,size),0)
        [dx,dy] = shift * np.array([-np.sin(theta), np.cos(theta)])
        # print(int(dx),int(dy))
        shadow = op*sii.shift(shadow, shift=[int(dx),int(dy)],mode='constant',cval=0)
        return shadow.astype('uint8')

    def border(self, alpha, size, kernel_type='ELLIPSE'):
        """
        alpha : alpha layer of the text
        size  : size of the kernel
        kernel_type : one of [rect,ellipse,cross]

        @return : alpha layer of the border (color to be added externally).
        """
        kdict = {'RECT':cv.MORPH_RECT, 'ELLIPSE':cv.MORPH_ELLIPSE,
                 'CROSS':cv.MORPH_CROSS}
        kernel = cv.getStructuringElement(kdict[kernel_type],(size,size))
        border = cv.dilate(alpha,kernel,iterations=1) # - alpha
        return border

    def blend(self,cf,cb,mode='normal'):
        return cf

    def merge_two(self,fore,back,blend_type=None):
        """
        merge two FOREground and BACKground layers.
        ref: https://en.wikipedia.org/wiki/Alpha_compositing
        ref: Chapter 7 (pg. 440 and pg. 444):
             http://partners.adobe.com/public/developer/en/pdf/PDFReference.pdf
        """
        a_f = fore.alpha/255.0
        a_b = back.alpha/255.0
        c_f = fore.color
        c_b = back.color

        a_r = a_f + a_b - a_f*a_b
        if blend_type != None:
            c_blend = self.blend(c_f, c_b, blend_type)
            c_r = (   ((1-a_f)*a_b)[:,:,None] * c_b
                    + ((1-a_b)*a_f)[:,:,None] * c_f
                    + (a_f*a_b)[:,:,None] * c_blend   )
        else:
            c_r = (   ((1-a_f)*a_b)[:,:,None] * c_b
                    + a_f[:,:,None]*c_f    )

        return Layer((255*a_r).astype('uint8'), c_r.astype('uint8'))

    def merge_down(self, layers, blends=None):
        """
        layers  : [l1,l2,...ln] : a list of LAYER objects.
                 l1 is on the top, ln is the bottom-most layer.
        blend   : the type of blend to use. Should be n-1.
                 use None for plain alpha blending.
        Note    : (1) it assumes that all the layers are of the SAME SIZE.
        @return : a single LAYER type object representing the merged-down image
        """
        nlayers = len(layers)
        if nlayers > 1:
            [n,m] = layers[0].alpha.shape[:2]
            out_layer = layers[-1]
            for i in range(-2,-nlayers-1,-1):
                blend=None
                if blends is not None:
                    blend = blends[i+1]
                    out_layer = self.merge_two(fore=layers[i], back=out_layer, blend_type=blend)
            return out_layer
        else:
            return layers[0]

    def resize_im(self, im, osize):
        return np.array(Image.fromarray(im).resize(osize[::-1], Image.BICUBIC))
        
    def occlude(self):
        """
        somehow add occlusion to text.
        """
        pass

    def complement(self, rgb_color):
        """
        return a color which is complementary to the RGB_COLOR.
        """
        col_hsv = np.squeeze(cv.cvtColor(rgb_color[None,None,:], cv.COLOR_RGB2HSV))
        col_hsv[0] = col_hsv[0] + 128 #uint8 mods to 255
        col_comp = np.squeeze(cv.cvtColor(col_hsv[None,None,:],cv.COLOR_HSV2RGB))
        return col_comp

    def triangle_color(self, col1, col2):
        """
        Returns a color which is "opposite" to both col1 and col2.
        """
        col1, col2 = np.array(col1), np.array(col2)
        col1 = np.squeeze(cv.cvtColor(col1[None,None,:], cv.COLOR_RGB2HSV))
        col2 = np.squeeze(cv.cvtColor(col2[None,None,:], cv.COLOR_RGB2HSV))
        h1, h2 = col1[0], col2[0]
        if h2 < h1 : h1,h2 = h2,h1 #swap
        dh = h2-h1
        if dh < 127: dh = 255-dh
        col1[0] = h1 + dh/2
        return np.squeeze(cv.cvtColor(col1[None,None,:],cv.COLOR_HSV2RGB))

    def color_border(self, col_text, col_bg):
        """
        Decide on a color for the border:
            - could be the same as text-color but lower/higher 'VALUE' component.
            - could be the same as bg-color but lower/higher 'VALUE'.
            - could be 'mid-way' color b/w text & bg colors.
        """
        choice = np.random.choice(3)

        col_text = cv.cvtColor(col_text, cv.COLOR_RGB2HSV)
        col_text = np.reshape(col_text, (np.prod(col_text.shape[:2]),3))
        col_text = np.mean(col_text,axis=0).astype('uint8')

        vs = np.linspace(0,1)
        def get_sample(x):
            ps = np.abs(vs - x/255.0)
            ps /= np.sum(ps)
            v_rand = np.clip(np.random.choice(vs,p=ps) + 0.1*np.random.randn(),0,1)
            return 255*v_rand

        # first choose a color, then inc/dec its VALUE:
        if choice==0:
            # increase/decrease saturation:
            col_text[0] = get_sample(col_text[0]) # saturation
            col_text = np.squeeze(cv.cvtColor(col_text[None,None,:],cv.COLOR_HSV2RGB))
        elif choice==1:
            # get the complementary color to text:
            col_text = np.squeeze(cv.cvtColor(col_text[None,None,:],cv.COLOR_HSV2RGB))
            col_text = self.complement(col_text)
        else:
            # choose a mid-way color:
            col_bg = cv.cvtColor(col_bg, cv.COLOR_RGB2HSV)
            col_bg = np.reshape(col_bg, (np.prod(col_bg.shape[:2]),3))
            col_bg = np.mean(col_bg,axis=0).astype('uint8')
            col_bg = np.squeeze(cv.cvtColor(col_bg[None,None,:],cv.COLOR_HSV2RGB))
            col_text = np.squeeze(cv.cvtColor(col_text[None,None,:],cv.COLOR_HSV2RGB))
            col_text = self.triangle_color(col_text,col_bg)

        # now change the VALUE channel:        
        col_text = np.squeeze(cv.cvtColor(col_text[None,None,:],cv.COLOR_RGB2HSV))
        col_text[2] = get_sample(col_text[2]) # value
        return np.squeeze(cv.cvtColor(col_text[None,None,:],cv.COLOR_HSV2RGB))


    def process(self, text_arr, bg_arr, text_info, ftype=0):
        """
        text_arr : one alpha mask : nxm, uint8
        bg_arr   : background image: nxmx3, uint8
        min_h    : height of the smallest character (px)

        return text_arr blit onto bg_arr.
        """
        # decide on a color for the text:
        l_text = Layer(alpha=text_arr, color=text_info["rgb"])
        bg_col = np.mean(np.mean(bg_arr,axis=0),axis=0)
        bg_col = (255,255,255)
        l_bg = Layer(alpha=255*np.ones_like(text_arr,'uint8'),color=bg_col)

        # if text_info["alpha"] == 1:
        #     l_text.alpha = l_text.alpha * np.clip(0.88 + 0.1*np.random.randn(), 0.72, 1.0)

        if text_info["alpha"] == 1:
            l_text.alpha = l_text.alpha * np.clip(0.95 + 0.1*np.random.randn(), 0.9, 1.0)
        elif text_info["alpha"] == 2:
            l_text.alpha = l_text.alpha * np.clip(0.85 + 0.1*np.random.randn(), 0.8, 0.9)
        elif text_info["alpha"] == 3:
            l_text.alpha = l_text.alpha * np.clip(0.75 + 0.1*np.random.randn(), 0.7, 0.8)
        elif text_info["alpha"] == 4:
            l_text.alpha = l_text.alpha * np.clip(0.65 + 0.1*np.random.randn(), 0.6, 0.7)

        layers = [l_text]
        blends = []

        # add stroke
        if text_info["stroke"]==1:
            border_a = self.border(l_text.alpha, size=text_info["stroke_width"]*2+1)
            # l_border = Layer(border_a, self.color_border(l_text.color,l_bg.color))
            l_border = Layer(border_a, text_info["stroke_fill"])

            l_bg = Layer(alpha=255*np.ones_like(text_arr,'uint8'), color=bg_arr)
            l_n = self.merge_down([l_border, l_bg], ["normal"])
            cv2.imwrite("pic/out.png", l_n.color) 

            layers.append(l_border)
            blends.append('normal')

        # add shadow:
        if text_info["shadow"]==1:
            # shadow gaussian size:
            bsz = text_info["shadow_gaussian"]
            # shadow angle:
            theta = text_info["shadow_theta"]
            # shadow shift:
            shift = text_info["shadow_shift"]
            # opacity:
            op = text_info["shadow_alpha"]
            if text_info["stroke"]==0:
                shadow = self.drop_shadow(l_text.alpha, theta, shift, bsz, op)
            else:
                shadow = self.drop_shadow(l_border.alpha, theta, shift, bsz, op)
            l_shadow = Layer(shadow, text_info["shadow_fill"])
            layers.append(l_shadow)
            blends.append('normal')
        
        if text_info["poisson"]:
            layers.append(l_bg)
            blends.append('normal')
            l_normal = self.merge_down(layers,blends)

            # now do poisson image editing:
            l_bg = Layer(alpha=255*np.ones_like(text_arr,'uint8'), color=bg_arr)
            layers[-1] = l_bg
            l_out = blit_images(l_normal.color,l_bg.color.copy())
        else:
            l_bg = Layer(alpha=255*np.ones_like(text_arr,'uint8'), color=bg_arr)
            layers.append(l_bg)
            blends.append('normal')
            l_out = self.merge_down(layers, blends).color
        # cv2.imwrite("pic/normal.png", l_normal.color) 
        # cv2.imwrite("pic/bg.png", l_bg.color) 
        # cv2.imwrite("pic/out.png", l_out) 
    
        return l_out

    def check_perceptible(self, txt_mask, bg, txt_bg):
        """
        --- DEPRECATED; USE GRADIENT CHECKING IN POISSON-RECONSTRUCT INSTEAD ---

        checks if the text after merging with background
        is still visible.
        txt_mask (hxw) : binary image of text -- 255 where text is present
                                                   0 elsewhere
        bg (hxwx3) : original background image WITHOUT any text.
        txt_bg (hxwx3) : image with text.
        """
        bgo,txto = bg.copy(), txt_bg.copy()
        txt_mask = txt_mask.astype('bool')
        bg = cv.cvtColor(bg.copy(), cv.COLOR_RGB2Lab)
        txt_bg = cv.cvtColor(txt_bg.copy(), cv.COLOR_RGB2Lab)
        bg_px = bg[txt_mask,:]
        txt_px = txt_bg[txt_mask,:]
        bg_px[:,0] *= 100.0/255.0 #rescale - L channel
        txt_px[:,0] *= 100.0/255.0

        diff = np.linalg.norm(bg_px-txt_px,ord=None,axis=1)
        diff = np.percentile(diff,[10,30,50,70,90])
        print ("color diff percentile :", diff)
        return diff, (bgo,txto)

    def color(self, bg_arr, text_arr, text_info, place_order=None, pad=20, ftype=0):
        """
        Return colorized text image.

        text_arr : list of (n x m) numpy text alpha mask (unit8).
        hs : list of minimum heights (scalar) of characters in each text-array. 
        text_loc : [row,column] : location of text in the canvas.
        canvas_sz : size of canvas image.
        
        return : nxmx3 rgb colorized text-image.
        """
        bg_arr = bg_arr.copy()
        if bg_arr.ndim == 2 or bg_arr.shape[2]==1: # grayscale image:
            bg_arr = np.repeat(bg_arr[:,:,None], 3, 2)

        # get the canvas size:
        canvas_sz = np.array(bg_arr.shape[:2])

        # initialize the placement order:
        if place_order is None:
            place_order = np.array(range(len(text_arr)))

        rendered = []
        for i in place_order[::-1]:
            # get the "location" of the text in the image:
            ## this is the minimum x and y coordinates of text:
            loc = np.where(text_arr[i])
            if loc[0].size == 0:
                return bg_arr
            lx, ly = np.min(loc[0]), np.min(loc[1])
            mx, my = np.max(loc[0]), np.max(loc[1])
            l = np.array([lx,ly])
            m = np.array([mx,my])-l+1
            text_patch = text_arr[i][l[0]:l[0]+m[0],l[1]:l[1]+m[1]]

            # figure out padding:
            ext = canvas_sz - (l+m)
            num_pad = pad*np.ones(4,dtype='int32')
            num_pad[:2] = np.minimum(num_pad[:2], l)
            num_pad[2:] = np.minimum(num_pad[2:], ext)
            text_patch = np.pad(text_patch, pad_width=((num_pad[0],num_pad[2]), (num_pad[1],num_pad[3])), mode='constant')
            l -= num_pad[:2]

            w,h = text_patch.shape
            bg = bg_arr[l[0]:l[0]+w,l[1]:l[1]+h,:]

            rdr0 = self.process(text_patch, bg, text_info, ftype)
            rendered.append(rdr0)

            bg_arr[l[0]:l[0]+w,l[1]:l[1]+h,:] = rdr0#rendered[-1]


            return bg_arr

        return bg_arr
