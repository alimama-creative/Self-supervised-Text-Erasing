from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
       
        self.parser.add_argument('--display_freq', type=int, default=200, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
        self.parser.add_argument('--valid_freq', type=int, default=800, help='frequency of saving the latest results')
        
        self.parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        self.parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        self.parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        
        self.parser.add_argument('--baseline', type=str, default="null", help='null, ohem, focal')

        self.parser.add_argument('--lambda1', type=float, default=1.0, help='the ratio of data to update controller')
        self.parser.add_argument('--lambda2', type=float, default=1.0, help='the ratio of data to update controller')
        self.parser.add_argument('--lambda3', type=float, default=1.0, help='the ratio of data to update controller')
        self.parser.add_argument('--lambda4', type=float, default=1.0, help='the ratio of data to update controller')
        self.parser.add_argument('--diff_range', type=int, default=2.0, help='the ratio of data to update controller')
        
        
        self.parser.add_argument('--ctl_M', type=int, default=20, help='number of strategy')
        self.parser.add_argument('--ctl_layer', type=int, default=1, help='the freq of update controller')
        self.parser.add_argument('--ctl_freq', type=int, default=100, help='the freq of update controller')
        self.parser.add_argument('--ctl_train_freq', type=int, default=20, help='the freq of update controller')
        self.parser.add_argument('--ctl_ratio', type=float, default=0.5, help='the ratio of data to update controller')
        self.parser.add_argument('--ctl_policy_num', type=int, default=20, help='the ratio of data to update controller')
        self.parser.add_argument('--ctl_update_num', type=int, default=50, help='the ratio of data to update controller')
        self.parser.add_argument('--ctl_batchSize', type=int, default=4, help='the ratio of data to update controller')
        
        self.parser.add_argument('--reward_norm', type=str, default="mean", help='mean, norm')
        self.parser.add_argument('--aux_reward', action='store_true', help='number of strategy')
        self.parser.add_argument('--aux_dataset', action='store_true', help='number of strategy')
        self.parser.add_argument('--continual_loss', action='store_true', help='number of strategy')
        self.parser.add_argument('--cont_weight', type=int, default=500, help='the freq of update controller')
        
        self.parser.add_argument('--sigmoid', action='store_true', help='train with only hole loss')
        self.parser.add_argument('--hole_loss', action='store_true', help='train with only hole loss')
        self.parser.add_argument('--style_loss', action='store_true', help='train with style loss')
        self.parser.add_argument('--contrastive_loss', action='store_true', help='train with style loss')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        self.parser.add_argument('--dbeta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--dbeta2', type=float, default=0.999, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--dlr', type=float, default=0.00001, help='initial learning rate for adam')
        self.parser.add_argument('--clr', type=float, default=0.0005, help='initial learning rate for adam')
        self.parser.add_argument('--d2g_lr', type=float, default=5, help='initial learning rate for adam')
        self.parser.add_argument('--gan_mode', type=str, default='vanilla', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        self.parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        self.parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        self.parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for L1 loss')
        self.parser.add_argument('--lambda_gan', type=float, default=1.0, help='weight for gan loss')
        self.parser.add_argument('--lambda_mask', type=float, default=10.0, help='weight for mask loss')
        self.parser.add_argument('--lambda_content', type=float, default=1.0, help='weight for content loss')
        self.parser.add_argument('--lambda_style', type=float, default=2.5, help='weight for style loss')

        self.isTrain = True
