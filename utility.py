
from data import CreateDataLoader
from util.util_list import mse_psnr
import copy
import random
import torch
import numpy as np


def valid_metrics_cal(valid_dataset, l, model, visualizer, epoch, verbose=True, iter=-1, m_type = 0):
    sum_mse, sum_psnr = 0, 0 
    sum_mse_m, sum_psnr_m = 0, 0 
    model.eval()
    for m, data in enumerate(valid_dataset):
        if m_type == 0:
            model.set_input(data)
        else:
            model.set_inputs(data)
            model.set_specific_image(0)
        model.test()
        res = model.get_current_image_tensor()
        for k in range(res["comp_B"].shape[0]):
            mse,psnr = mse_psnr(res["comp_B"][k], res["real_B"][k])
            sum_mse += mse
            sum_psnr += psnr
            mse,psnr = mse_psnr(res["comp_G"][k], res["real_B"][k])
            sum_mse_m += mse
            sum_psnr_m += psnr
    if verbose:
        if iter == -1:
            visualizer.print_valid_metric(epoch, sum_mse/l, sum_psnr/l)
        else:
            visualizer.print_valid_metric(epoch, sum_mse/l, sum_psnr/l, iter)
    model.train()
    return sum_mse/l, sum_psnr/l, sum_mse_m/l, sum_psnr_m/l


def init_dataset(opt, dist, batchSize=None, valid=None, serial_batches=None, aux=None,real_val=None):
    opt = copy.deepcopy(opt)
    if batchSize:
        opt.batchSize = batchSize
    if valid:
        opt.valid = valid
        valid = True
    if serial_batches:
        opt.serial_batches = serial_batches
    if aux:
        opt.aux_dataset = aux
    if real_val:
        opt.real_val = True
        opt.domain_in = False
    data_loader = CreateDataLoader(opt, dist, valid)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    return dataset, dataset_size


def random_seed_initial(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_data(datap, dataset, gen_configs):
    datas = []
    for i in range(len(datap["ind"])):
        datas.append(dataset.dataset.gen_data(datap["ind"][i], [gen_configs[i]]))
    data = {}
    data['img'] = torch.stack([d['img'] for d in datas], dim = 0)
    data['gt'] = torch.stack([d['gt'] for d in datas], dim = 0)
    data['raw_mask'] = torch.stack([d['raw_mask'] for d in datas], dim = 0)
    data['mask'] = torch.stack([d['mask'] for d in datas], dim = 0)
    data['path'] = [d['path'] for d in datas]
    # print("collan: ", end-begin)
    return data