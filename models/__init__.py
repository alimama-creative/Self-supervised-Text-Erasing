def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'pix2pix' or opt.model == 'Pix2pix':
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()       
    elif opt.model == 'disc':
        from .disc_model import DiscModel
        model = DiscModel() 
    # elif opt.model == 'erase_fix':
    #     from .erase_model_re import EraseModel
    #     model = EraseModel()       
    elif opt.model == 'gateconv':
        from .gateconv_model import GatedConvModel
        model = GatedConvModel()
    elif opt.model == 'erase' or opt.model == 'Erase' or opt.model == 'erasenet':
        from .erase_model import EraseModel
        model = EraseModel()   
    # elif opt.model == 'mtrnet':
    #     from .mtrnet_model import MTRNetModel
    #     model = MTRNetModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
