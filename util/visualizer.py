import numpy as np
import os
import ntpath
import time
from . import util_list
from . import html_util as html
from io import BytesIO
import json

def save_images(webpage, visuals, image_path):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    # print(short_path)
    name = os.path.splitext(short_path)[0]
    webpage.add_header(name)
    ims = []
    txts = []
    links = []

    for label, image_numpy in visuals.items():
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util_list.save_image(image_numpy, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=256)

class Visualizer():
    def __init__(self, opt, saver=None, dist=None):
        # self.opt = opt
        self.use_html = opt.isTrain and not opt.no_html
        self.name = opt.name
        self.dist = dist
        self.opt = opt
        self.saver = saver
        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.model, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util_list.mkdirs([self.web_dir, self.img_dir])

        if not opt.saveOnline:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.model, opt.name, 'loss_log.txt')
            self.valid_log_name = os.path.join(opt.checkpoints_dir, opt.model, opt.name, 'valid_loss_log.txt')
            self.policy_log_name = os.path.join(opt.checkpoints_dir, opt.model, opt.name, 'policy_log.txt')
            now = time.strftime("%c")
            with open(self.log_name, "w") as log_file:
                log_file.write('================ Training Loss (%s) ================\n' % now)
            with open(self.valid_log_name, "w") as log_file:
                log_file.write('================ Validation Loss (%s) ================\n' % now)
        save = True
        if self.opt.online :
            if self.dist.get_rank()!=0:
                save = False
        if self.opt.saveOnline and save:
            self.log_name = os.path.join("gangwei.jgw/models/", opt.model, opt.name, 'log', 'loss_log.txt')
            self.valid_log_name = os.path.join("gangwei.jgw/models/", opt.model, opt.name, 'log', 'valid_loss_log.txt')
            self.policy_log_name = os.path.join("gangwei.jgw/models/", opt.model, opt.name, 'log', 'policy_log.txt')
            now = time.strftime("%c")
            self.log_io = BytesIO()
            self.log_io.write(b'================ Training Loss (%b) ================\n' % bytes(now, encoding = "utf8"))
            saver.save_log(self.log_name, self.log_io)
            # saver.bucket.put_object(self.log_name, self.log_io.getvalue())
            self.valid_log_io = BytesIO()
            self.valid_log_io.write(b'================ Validation Loss (%b) ================\n' % bytes(now, encoding = "utf8"))
            saver.save_log(self.valid_log_name, self.valid_log_io)
            # saver.bucket.put_object(self.valid_log_name, self.valid_log_io.getvalue())
            self.policy_log_io = BytesIO()

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, iter=0):
        if not self.opt.saveOnline:
            if self.use_html: # save images to a html file
                for label, image_numpy in visuals.items():
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                    util_list.save_image(image_numpy, img_path)

                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%.3d_%s.png' % (epoch, iter, label))
                    util_list.save_image(image_numpy, img_path)
                # update website
                webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
                for n in range(epoch, 0, -1):
                    webpage.add_header('epoch [%d]' % n)
                    ims = []
                    txts = []
                    links = []

                    for label, image_numpy in visuals.items():
                        img_path = 'epoch%.3d_%s.png' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                    webpage.add_images(ims, txts, links, width=256)
                webpage.save()
        save = True
        if self.opt.online :
            if self.dist.get_rank()!=0:
                save = False
        if self.opt.saveOnline and save:
            for label, image_numpy in visuals.items():
                self.saver.save_image(epoch, "%.3d_%s"%(iter, label), image_numpy)
                self.saver.save_image(epoch, label, image_numpy)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        curr_time = time.strftime("%b-%d-%H:%M:%S", time.localtime())
        message = '%s (epoch: %d, iters: %d, time: %.3f) ' % (curr_time, epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)
        print(message)
        if not self.opt.saveOnline:
            with open(self.log_name, "a") as log_file:
                log_file.write('%s\n' % message)
        save = True
        if self.opt.online :
            if self.dist.get_rank()!=0:
                save = False
        if self.opt.saveOnline and save:
            self.log_io.write(b'%b\n' % bytes(message, encoding = "utf8"))
            self.saver.save_log(self.log_name, self.log_io)
            # bucket.put_object(self.log_name, self.log_io.getvalue())

    # errors: same format as |errors| of plotCurrentErrors
    def print_valid_metric(self, epoch, mse, psnr, iter=-1):
        if iter == -1:
            message = '(epoch: %d) mse: %.6f psnr: %.3f ' % (epoch, mse, psnr)
        else:
            message = '(epoch: %d iter %d) mse: %.6f psnr: %.3f ' % (epoch, iter, mse, psnr)
        print(message)
        if not self.opt.saveOnline:
            with open(self.log_name, "a") as log_file:
                log_file.write('%s\n' % message)
            with open(self.valid_log_name, "a") as log_file:
                log_file.write('%s\n' % message)
        save = True
        if self.opt.online :
            if self.dist.get_rank()!=0:
                save = False
        if self.opt.saveOnline and save:
            self.log_io.write(b'%b\n' % bytes(message, encoding = "utf8"))
            self.saver.save_log(self.log_name, self.log_io)
            # self.saver.bucket.put_object(self.log_name, self.log_io.getvalue())

            self.valid_log_io.write(b'%b\n' % bytes(message, encoding = "utf8"))
            self.saver.save_log(self.valid_log_name, self.valid_log_io)
            # self.saver.bucket.put_object(self.valid_log_name, self.valid_log_io.getvalue())
    
    def print_valid_metric_list(self, epoch, psnr, iter=-1):
        if iter == -1:
            message = '(epoch: %d) psnr: ' % (epoch)
        else:
            message = '(epoch: %d iter %d) psnr: ' % (epoch, iter)
        for p in psnr:
            message += ' %.3f;' % p
        print(message)
        if not self.opt.saveOnline:
            with open(self.log_name, "a") as log_file:
                log_file.write('%s\n' % message)
            with open(self.valid_log_name, "a") as log_file:
                log_file.write('%s\n' % message)
        save = True
        if self.opt.online :
            if self.dist.get_rank()!=0:
                save = False
        if self.opt.saveOnline and save:
            self.log_io.write(b'%b\n' % bytes(message, encoding = "utf8"))
            self.saver.save_log(self.log_name, self.log_io)
            # self.saver.bucket.put_object(self.log_name, self.log_io.getvalue())

            self.valid_log_io.write(b'%b\n' % bytes(message, encoding = "utf8"))
            self.saver.save_log(self.valid_log_name, self.valid_log_io)

    def print_policy_log(self, epoch, configs):
        message = json.dumps(configs)
        message = "epoch %3d: %s" % (epoch, message)
        if not self.opt.saveOnline:
            with open(self.policy_log_name, "a") as log_file:
                log_file.write('%s\n' % message)
        save = True
        if self.opt.online :
            if self.dist.get_rank()!=0:
                save = False
        if self.opt.saveOnline and save:
            self.policy_log_io.write(b'%b\n' % bytes(message, encoding = "utf8"))
            self.saver.save_log(self.policy_log_name, self.policy_log_io)


    # save image to the dis
    def save_images_demo(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = (image_path[0]).split("/")[-1]
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s.jpg' % (name)
            save_path = os.path.join(image_dir, image_name)
            util_list.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=256)
