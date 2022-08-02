import enum
from cv2 import PARAM_UNSIGNED_INT
import torch
import torch.distributed as dist
import numpy as np
import random
import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from models.controller import Controller
from models.networks import init_net
from util.visualizer import Visualizer
from util.util_list import  gen_config_from_parse
from utility import get_data, random_seed_initial, init_dataset, valid_metrics_cal
import os
import pygame
os.environ['NCCL_IB_DISABLE'] = '1'


def main():
    # initial process
    opt = TrainOptions().parse()

    pygame.init()
    # initialize random seed
    random_seed_initial(opt.seed)

    if opt.model in ['Erase', 'erasenet', 'erase', 'gateconv']:
        opt.data_norm = False

    opt.real_val = False
    realistic_reward = False
    realistic_val_reward = False
    difficult_reward = False

    if "1" in opt.reward_type:
        realistic_reward = True
        opt.netD_M = True
    if "2" in opt.reward_type:
        difficult_reward = True
    if "4" in opt.reward_type:
        realistic_val_reward = True
        opt.netD_M = True
    reward_norm = opt.reward_norm

    # data loader initial process'
    dataset, dataset_size = init_dataset(opt, dist)
    print('#training images = %d' % dataset_size)
    ctl_dataset, _ = init_dataset(opt, dist, batchSize=opt.ctl_batchSize)
    ctl_dataset.dataset.lant = True
    ctl_dataset_iter = iter(ctl_dataset)
    # valid data loader initial process
    if opt.valid == 1:
        valid_dataset, v_l = init_dataset(opt, dist, batchSize=4, valid=3, serial_batches=True)
        print('#valid images = %d' % v_l)
        test_dataset, t_l = init_dataset(opt, dist, batchSize=4, valid=2, serial_batches=True)
        print('#test images = %d' % t_l)
    real_dataset, r_l = init_dataset(opt, dist, real_val=True)
    print('#real train images = %d' % r_l)

    # model initial
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt, dist)

    # controller initial
    # if opt.online:
    #     controller = Controller(layers=opt.ctl_layer, gen_space=opt.gen_space, is_cuda=True)
    #     controller_copy = Controller(layers=opt.ctl_layer, gen_space=opt.gen_space)
    #     # controller = init_net(controller, opt.init_type, opt.init_gain, opt.gpu_ids, opt.online)
    # else:
    controller = Controller(layers=opt.ctl_layer, gen_space=opt.gen_space, is_cuda=False)
    
    controller = init_net(controller, opt.init_type, opt.init_gain, [], opt.online)
    controller_optimizer = torch.optim.Adam(controller.parameters(), lr = opt.clr)
    # te1
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu') 
    if opt.continue_train and not opt.adg_start:        
        load_filename = '%s_net_Controller.pth' % str(epoch)
        load_path = os.path.join(opt.checkpoints_dir, opt.model, opt.name, load_filename)
        state_dict = torch.load(load_path, map_location=str(device))
        controller.load_state_dict(state_dict)

    # begin epoch set
    total_steps = 0
    if opt.continue_train:
        begin_epoch = int(opt.which_epoch) + 1
    else:
        begin_epoch = 1
    
    best_metrics = 0
    stop_epoch = 0

    M = opt.ctl_M
    controller_update = opt.ctl_freq
    # training begin
    # for epoch in [20,30,40,50]:
        # opt.which_epoch = epoch
        # model.setup(opt, saver)
        # print("epoch: ", epoch)
    if opt.valid != 0 and opt.continue_train:
        _,psnr_v,_,psnr_v_m = valid_metrics_cal(valid_dataset, v_l, model, visualizer, begin_epoch-1, False, m_type=1)
        _,psnr_t,_,psnr_t_m = valid_metrics_cal(test_dataset, t_l, model, visualizer, begin_epoch-1, False, m_type=1)
        visualizer.print_valid_metric_list(begin_epoch-1, [psnr_v, psnr_t, psnr_v_m, psnr_t_m])
    
    for epoch in range(begin_epoch):
        model.update_learning_rate() 

    if difficult_reward:
        Lm_mean = torch.zeros(1, requires_grad=False).cuda()
        lambda1 = opt.lambda1
    if realistic_reward or realistic_val_reward:
        Ln_mean = torch.zeros(1, requires_grad=False).cuda() * 0.5
        lambda2 = opt.lambda2

    ctl_cnt = 0
    policy_cnt = np.zeros([30,12])
    for epoch in range(begin_epoch, opt.n_epochs + opt.n_epochs_decay + 1):
        # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate() 
        epoch_start_time = time.time()
        iter_start_time = time.time()
        model.train()
        controller.train()
        cost_time = [0 for _ in range(6)]
        # if opt.online:
        #     dataset.sampler.set_epoch(opt.seed+epoch)
        #     for weights, cp_weights in zip(controller.module.parameters(), controller_copy.parameters()):
        #         cp_weights.data.copy_(weights.data)
        #     dataset.dataset.controller = controller_copy
        # else:          
        dataset.dataset.controller = controller
        # dataset_iter = iter(real_dataset)
        dataset_iter = iter(dataset)
        # prefecher.reset()
        reward, rvar = [], []
        for i in range(len(dataset)):
            # update generate policies
            if realistic_val_reward and i%200==0:
                # if opt.online:
                #     real_dataset.sampler.set_epoch(opt.seed+i+epoch)
                #     for weights, cp_weights in zip(controller.module.parameters(), controller_copy.parameters()):
                #         cp_weights.data.copy_(weights.data)
                #     real_dataset.dataset.controller = controller_copy
                # else:          
                real_dataset.dataset.controller = controller
                # for _ in range(2):
                for _ in range(1):
                    for data in real_dataset:
                        model.set_inputs(data)
                        model.set_specific_image(0)
                        model.forward()
                        model.optimize_mask_dis()
            
            cost_time[0] -= time.time()
            data = next(dataset_iter)
            model.set_inputs(data)
            model.set_specific_image(0)
            cost_time[0] += time.time()

            cost_time[1] -= time.time()
            total_steps += 1
            model.train()
            if opt.baseline == "dann":
                model.optimize_parameters(opt.baseline)
            else:
                model.optimize_parameters()
            # model.forward()
            # model.optimize_mask_dis()
            cost_time[1] += time.time()

            # controller update
            cost_time[2] -= time.time()
            lo_m = 0
            info = [[0,1,20],[10,11,20],[11,12,14],[14,15,20]]
            if (i+1) % controller_update == 0:
                print("ctl updating .........")
                model.eval()
                ctl_batch_num = max(1, int(opt.ctl_update_num/opt.ctl_batchSize))
                ctl_cnt += ctl_batch_num
                if ctl_cnt >= len(ctl_dataset):
                    # if opt.online:
                    #     ctl_dataset.sampler.set_epoch(opt.seed+i+epoch)
                    ctl_dataset_iter = iter(ctl_dataset)
                    ctl_cnt = ctl_batch_num
                for j in range(ctl_batch_num):
                    datap = next(ctl_dataset_iter)
                    # if opt.online:
                    #     policies, log_probs, entropies = controller(x=datap["latent"].cuda(), verbose=False)
                    # else:
                    policies, log_probs, entropies = controller(x=datap["latent"], verbose=False)
                    policies = policies.cpu().detach().numpy()
                #     for x in range(policies.shape[0]):
                #         policy_cnt[0,-1] += 1
                #         f = True
                #         for y in range(policies.shape[1]):
                #             for z in range(len(info)):
                #                 if policies[x,info[z][0]] == 0 and y >= info[z][1] and y <= info[z][2]:
                #                     f = False
                #             if f:
                #                 policy_cnt[y,policies[x,y]] += 1
                #             f = True
                # print(policy_cnt/policy_cnt[0,-1])
                # exit()
                    gen_configs = gen_config_from_parse(policies, opt.gen_space)
                    data = get_data(datap, ctl_dataset, gen_configs)
                    model.set_inputs(data)
                    model.set_specific_image(0)
                    model.forward()
                    Ls = torch.zeros(len(data['path']), requires_grad=False).cuda()
                    if difficult_reward:
                        Lm = model.calcu_loss().detach()

                        lo_m += torch.sum(Lm)
                        # print(Lm.shape)
                        if Lm_mean == 0:
                            Lm_mean = torch.mean(Lm)
                        Lm_mean = 0.95 * Lm_mean + 0.05 * torch.mean(Lm)
                        if reward_norm == "mean":
                            Lm = (Lm - Lm_mean) * 60
                        elif reward_norm == "exp":
                            # Lm = -torch.abs(1-torch.exp((Lm - 3*Lm_mean)*300))
                            Lm = 5*(1-torch.abs(1-torch.exp((Lm - opt.diff_range*Lm_mean)*20)))
                        elif reward_norm == "norm":
                            # Lm = 5*(1-torch.abs(1-torch.exp((Lm - opt.diff_range*Lm_mean)*20)))
                            # Lm = (Lm - Lm_mean) * 100
                            Lm = (Lm - torch.mean(Lm))/(torch.std(Lm) + 1e-5)
                        Ls += lambda1 * Lm
                        # print("diffcult reward:", Lm_mean, lambda1 * Lm)
                    if realistic_reward or realistic_val_reward:
                        Ln = model.calcu_real_reward().detach() 
                        lo_m -= torch.sum(Ln)
                        Ln_mean = 0.95 * Ln_mean + 0.05 * torch.mean(Ln)
                        if reward_norm == "norm":
                            Ln = -(Ln - torch.mean(Ln))/(torch.std(Ln) + 1e-5)
                        else:
                            Ln = -10 * (Ln - Ln_mean)
                        Ls += lambda2 * Ln
                        # print("realistic reward:", Ln_mean, lambda2 * Ln)

                    # if not opt.online:
                    Ls = Ls.cpu()
                    # print("total reward: ", Ls)
                    controller_optimizer.zero_grad()
                    # print(j, Ls, -log_probs)
                    score_loss = torch.mean(-log_probs * Ls) # - derivative of Score function
                    entropy_penalty = torch.mean(entropies) # Entropy penalty
                    controller_loss = score_loss - 1e-5 * entropy_penalty
                    # print("loss: ", score_loss, entropy_penalty, controller_loss)
                    controller_loss.backward()
                    # if opt.online:
                    #     controller.module.cnt += 1
                    # controller.print_grad()
                    controller_optimizer.step()
                #te1
                #     if j % 50 == 0:
                #         print(i, j, "iter valid......, current batch reward: ", lo_m)
                #         lo_m = 0
                #         real_dataset.dataset.controller = controller
                #         pols = torch.empty(0)
                #         for k, data in enumerate(real_dataset):
                #             if k>5:
                #                 break
                #             model.set_inputs(data)
                #             model.set_specific_image(0)
                #             model.forward()
                #             Ls = torch.zeros(len(data['path']), requires_grad=False).cuda()
                #             if difficult_reward:
                #                 Lm = model.calcu_loss().detach()
                #                 lo_m += torch.sum(Lm)
                #             if realistic_reward or realistic_val_reward:
                #                 Ln = model.calcu_real_reward().detach() 
                #                 lo_m -= torch.sum(Ln)
                #             pols = torch.cat((pols, data["policy"]))
                #         pols = pols/(torch.Tensor(controller.num_size)-1)
                #         # print(pols.shape)
                #         v = torch.sum(torch.var(pols,dim=0))
                #         print("valid reward: ", lo_m, "var", v)
                #         reward.append(lo_m.tolist())
                #         rvar.append(v.tolist())
                # print("reward record: ", reward)
                # print("var record: ", rvar)
                # if opt.online:
                #     for weights, cp_weights in zip(controller.module.parameters(), controller_copy.parameters()):
                #         cp_weights.data.copy_(weights.data)
                #     dataset.dataset.controller = controller_copy
                # else:          
                dataset.dataset.controller = controller
            cost_time[2] += time.time()

            # log print
            if total_steps % opt.display_freq == 0:
                visualizer.display_current_results(model.get_current_visuals(), epoch, int(i/100))
                
            if total_steps % opt.print_freq == 0:
                t = (time.time() - iter_start_time)
                iter_start_time = time.time()
                visualizer.print_current_errors(epoch, total_steps, model.get_current_losses(), t)
                print(cost_time)
                cost_time = [0 for _ in range(6)]

            if ((i+1) % opt.valid_freq == 0 or i == len(dataset)-1) and opt.valid != 0:
                _,psnr_v,_,psnr_v_m = valid_metrics_cal(valid_dataset, v_l, model, visualizer, epoch, False, m_type=1)
                _,psnr_t,_,psnr_t_m = valid_metrics_cal(test_dataset, t_l, model, visualizer, epoch, False, m_type=1)
                if i == len(dataset)-1:
                    visualizer.print_valid_metric_list(epoch, [psnr_v, psnr_t, psnr_v_m, psnr_t_m])
                else:
                    visualizer.print_valid_metric_list(epoch, [psnr_v, psnr_t, psnr_v_m, psnr_t_m], i)
                if i == len(dataset)-1:
                    if "ens" in opt.dataset_mode:
                        interal = 500
                    else:
                        interal = 50
                    if epoch > begin_epoch+interal:
                        stop_epoch += 1
                                
                if psnr_v_m > best_metrics:
                    best_metrics = psnr_v_m
                    stop_epoch = 0
                    save_path = os.path.join(model.save_dir, 'best_net_Controller.pth')
                    torch.save(controller.state_dict(), save_path)
                    model.save_networks('best')
        
        save_path = os.path.join(model.save_dir, 'latest_net_Controller.pth')
        torch.save(controller.state_dict(), save_path)
        model.save_networks('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks(epoch)
            save_path = os.path.join(model.save_dir, '%s_net_Controller.pth'%epoch)
            torch.save(controller.state_dict(), save_path)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        if stop_epoch >4:
            return

if __name__ == "__main__":
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)
    main()

