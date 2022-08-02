
# Self-supervised text erasing

## Dataset

The example datasets are provided in './examples' . And the entire datasets can be found as follows:

1. PosterErase

    You can download our poster text datatset at [PosterErase](https://tianchi.aliyun.com/dataset/dataDetail?dataId=134810#1).

2. SceneErase

    You can download the scene text datatset at [SCUT-EnsText](https://github.com/HCIILAB/SCUT-EnsText). And the extended unlabeled scene text dataset can be found at [ICDAR2019-ArT](https://ai.baidu.com/broad/download?dataset=art).

## Model

You can download the checkpoints for the posterErase dataset as follows:

1. Self-supervised Text Trasing : 
    
    download from [oss](), and put it under './checkpoints/erasenet/ste/best_net_G.pth'

2. Finetuning on STE : 

    download from [oss](), and put it under './checkpoints/erasenet/ft/best_net_G.pth'

## Runing

1. test 

    ```bash
    python test.py --dataset_mode items --dataroot ./examples/poster --model erasenet --name ft --which_epoch best  # inferece with the ste model on poster 

    python test.py --dataset_mode items --dataroot ./examples/poster --model erasenet --name ste --which_epoch best # inferece with the finetuned model model on poster  
    ```

2. train offline
    ```bash
    python train.py --batchSize 2 --dataset_mode items --dataroot ./examples/poster --model erasenet --valid 1  --gen_space random8 --name item-rand-s8-3 --PasteImage --sigmoid     # train with random synthesis policy on poster 

    python train.py --batchSize 2 --dataset_mode ens --dataroot ./examples/scene --model erasenet --valid 1  --gen_space random8 --name ens-rand-s8-3 --PasteImage --mask_sigmoid --sigmoid # train with random synthesis policy on erase 
    ```

3. train online with policy update
    ```bash
    python train_adg.py --batchSize 2 --dataset_mode items_adg --dataroot ./examples/poster  --model erasenet  --name item-sp8-r24 --gen_space random8 --lambda1 1 --lambda3 0 --lambda2 5 --reward_norm exp --reward_type 24 --valid 1 --ctl_M 3 --ctl_freq 10 --ctl_policy_num 2 --ctl_train_freq 10 --ctl_batchSize 2 --ctl_update_num 6  --PasteImage --sigmoid  # train with self-supervised synthesis policy on poster 

    python train_adg.py --batchSize 2 --dataset_mode ens_adg --dataroot ./examples/scene  --model erasenet  --name ens-sp8-r24 --gen_space random8 --lambda1 1 --lambda3 0 --lambda2 20 --reward_norm exp --reward_type 24 --valid 1 --ctl_M 3 --ctl_freq 10 --ctl_policy_num 2 --ctl_train_freq 10 --ctl_batchSize 2 --ctl_update_num 6  --PasteImage --mask_sigmoid --sigmoid # train with self-supervised synthesis policy on erase 
    ```

4. train online with policy outputed by feature
    ```bash
    python train_adg_data.py --batchSize 2 --dataset_mode items_adg_fea --dataroot ./examples/poster  --model erasenet --gen_space random8 --name item-sp8-data-r24 --clr 0.00005  --reward_norm exp --ctl_layer 2 --reward_type 24 --lambda1 1 --lambda2 10  --valid 1  --ctl_freq 10  --ctl_batchSize 2 --ctl_update_num 10 --PasteImage --sigmoid # train with self-supervised synthesis policy on poster 

    python train_adg_data.py --batchSize 2 --dataset_mode ens_adg_fea --dataroot ./examples/scene  --model erasenet --gen_space random8 --name ens-sp8-data-r24 --clr 0.00005  --reward_norm exp --ctl_layer 2 --reward_type 24 --lambda1 1 --lambda2 10  --valid 1  --ctl_freq 10  --ctl_batchSize 2 --ctl_update_num 10 --PasteImage --mask_sigmoid --sigmoid # train with self-supervised synthesis policy on erase 
    ```



### 预处理json文件内容
```
./examples/poster/train.txt
{
    "mask": "x1,x2,y1,y2;x1,x2,y1,y2;... ..." ;
    "place": {
        "obj": "text,size,direction;text,size,direction;... ..." ;
        <!-- "raw": "x1,x2,y1,y2;x1,x2,y1,y2;... ..." ; -->
        "text": [
            [
                [x,y,["c1,c2,c3",... ...]], 
                [x,y,["c1,c2,c3",... ...]], 
                ... ... 
            ],
            [
                [x,y,["c1,c2,c3",... ...]], 
                [x,y,["c1,c2,c3",... ...]], 
                ... ... 
            ], ... ...
        ]
    }
}
```

## Acknowledge
The repository is benefit a lot from [EraseNet](https://github.com/lcy0604/EraseNet.git) and [GatedConv](https://github.com/avalonstrel/GatedConvolution_pytorch.git). Thanks a lot for their excellent work.