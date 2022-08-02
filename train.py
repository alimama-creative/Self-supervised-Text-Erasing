import torch
import torch.distributed as dist
import time
from options.train_options import TrainOptions
from models import create_model
from utility import random_seed_initial, init_dataset, valid_metrics_cal
from util.visualizer import Visualizer
import os
import pygame

os.environ['NCCL_IB_DISABLE'] = '1'

def main():
    opt = TrainOptions().parse()

    pygame.init()
    random_seed_initial(opt.seed)
    
    if opt.model in ['Erase', 'erasenet', 'erase', 'gateconv']:
        opt.data_norm = False
    if opt.baseline == "domain":
        opt.domain_in = True

    dataset, dataset_size = init_dataset(opt, dist)
    print('#training images = %d' % dataset_size)
    if opt.valid == 1:
        valid_dataset, v_l = init_dataset(opt, dist, batchSize=4, valid=3, serial_batches=True)
        print('#valid images = %d' % v_l)
        test_dataset, t_l = init_dataset(opt, dist, batchSize=4, valid=2, serial_batches=True)
        print('#test images = %d' % t_l)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    
    cost = [0 for _ in range(5)]
    total_steps = 0
    if opt.continue_train:
        try:
            begin_epoch = int(opt.which_epoch) + 1
        except:
            begin_epoch = 1
    else:
        begin_epoch = 1
    
    best_metrics = 0
    save_best = False
    stop_epoch = 0

    # for epoch in [3000]:
    #     opt.which_epoch = epoch
    #     model.setup(opt, saver)
    #     print("epoch: ", epoch)
    if opt.valid != 0 and opt.continue_train:
        _,psnr_v,_,psnr_v_m = valid_metrics_cal(valid_dataset, v_l, model, visualizer, begin_epoch-1, False)
        _,psnr_t,_,psnr_t_m = valid_metrics_cal(test_dataset, t_l, model, visualizer, begin_epoch-1, False)
        visualizer.print_valid_metric_list(begin_epoch-1, [psnr_v, psnr_t, psnr_v_m, psnr_t_m])
    
    for epoch in range(begin_epoch):
        model.update_learning_rate() 

    cnt = 0
    for epoch in range(begin_epoch, opt.n_epochs + opt.n_epochs_decay + 1):
        model.update_learning_rate() 
        epoch_start_time = time.time()
        iter_start_time = time.time()
        model.train()            
        dataset_iter = iter(dataset)
        
        for i in range(len(dataset)):
            data = next(dataset_iter)
            total_steps += 1
            cost[0] -= time.time()
            model.set_input(data)
            cost[0] += time.time()
            cost[1] -= time.time()

            if opt.baseline == "domain":
                model.set_basic_input(data)
                model.forward_basic()
                model.optimize_parameters("domain")
            elif opt.baseline == "afn" or opt.baseline == "dann":
                model.optimize_parameters(opt.baseline)
            else:
                model.optimize_parameters()
            cost[1] += time.time()

            if total_steps % opt.display_freq == 0:
                visualizer.display_current_results(model.get_current_visuals(), epoch, int(i/100))

            if total_steps % opt.print_freq == 0:
                t = (time.time() - iter_start_time)
                iter_start_time = time.time()
                visualizer.print_current_errors(epoch, total_steps, model.get_current_losses(), t)
                # print(cost)
                cost = [0 for _ in range(5)]

            if ((i+1) % opt.valid_freq == 0 or i == len(dataset)-1) and opt.valid != 0:
                _,psnr_v,_,psnr_v_m = valid_metrics_cal(valid_dataset, v_l, model, visualizer, epoch, False)
                _,psnr_t,_,psnr_t_m = valid_metrics_cal(test_dataset, t_l, model, visualizer, epoch, False)
                if i == len(dataset)-1:
                    visualizer.print_valid_metric_list(epoch, [psnr_v, psnr_t, psnr_v_m, psnr_t_m])
                else:
                    visualizer.print_valid_metric_list(epoch, [psnr_v, psnr_t, psnr_v_m, psnr_t_m], i)
                if i == len(dataset)-1:
                    if "ens" in opt.dataset_mode:
                        interal = 500
                    else:
                        interal = 30
                    if epoch > begin_epoch+interal:
                        stop_epoch += 1
                if psnr_v_m > best_metrics:
                    best_metrics = psnr_v_m
                    stop_epoch = 0
                    model.save_networks('best')
        model.save_networks('latest')
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        if stop_epoch > 4:
            return

if __name__ == "__main__":
    main()