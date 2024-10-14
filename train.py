import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager, InfoNCELoss, augment_dec
from model import Model
from val import validation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    opt.phase = "train"

    if opt.language == 'En':
        from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
        train_dataset = Batch_Balanced_Dataset(opt)
    else:
        from dataset_zhiyu_blur import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
        train_dataset = Batch_Balanced_Dataset(opt)
        
    totalsample = train_dataset.get_total_training_sample()

    log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(opt.character)
        else:
            converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
        
    model = Model(opt).to(device)
    # model = torch.nn.DataParallel(model).to(device)   
    
    # Augment = augment_dec(device)    
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    # model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    print("Model:")
    print(model)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            # need to install warpctc. see our guideline.
            from warpctc_pytorch import CTCLoss 
            criterion = CTCLoss()
        else:
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
        
    #set ce loss for infonceloss
    CELoss   = torch.nn.CrossEntropyLoss().to(device) 
    lamb     = 0.1 # set scale factor for balance between the STR object and CL object     
    
    # loss averager
    loss_avg = Averager()
    
    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.adam:
        if opt.cycle_lr:
            max_lr =  opt.max_lr
            base_lr = opt.base_lr
            momentum = 0.9
            weight_decay = 0.00001
            total_iters = int(totalsample/opt.batch_size)
            print(total_iters)
            step_size_up =  int (total_iters*opt.step_size_up_ratio)
            step_size_down = total_iters - step_size_up
            optimizer = optim.Adam(filtered_parameters,lr=max_lr,weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=base_lr,max_lr=max_lr,step_size_up=step_size_up,step_size_down=step_size_down,mode="triangular2",cycle_momentum=False)
            with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
                opt_file.write("max_lr:" + str(max_lr)+"\n")
                opt_file.write("base_lr:" + str(base_lr)+"\n")
                opt_file.write("step_size_up:" + str(step_size_up)+"\n")
                opt_file.write("step_size_down:" + str(step_size_down)+"\n")
        else:
            optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=0.0005)
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps,weight_decay=0.0005)

    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

        
    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass
    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter
    for epoch in range(opt.max_epoch):
        with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
            log.write("epoch is: "+ str(epoch)+"\n")
            log.write("learning rate is: "+ str(optimizer.state_dict()['param_groups'][0]['lr'])+"\n")
        opt.phase = "train"
        for iter in range(train_dataset.total_iters()):
            iteration = iter
                
            # train part
            image_tensors, labels = train_dataset.get_batch()
            view1 = image_tensors.to(device)  
            text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
            batch_size   = image_tensors.size(0) 
                 
            if opt.CL:
                view2, blur_label2  = Augment(view1)   
        
            if opt.multi_task: 
                blabel1 = torch.tensor([0]).repeat(batch_size).to(device) #clear img idx is 0
                blabel2 = blur_label2.repeat(batch_size).to(device)
  
            if 'CTC' in opt.Prediction:    
                # forward
                preds1, z1, idx1 = model(view1, text)   
                preds_size = torch.IntTensor([preds1.size(1)] * batch_size) 
                # compute OCR loss
                if opt.baiduCTC:
                    preds1 = preds1.permute(1, 0, 2)  # to use CTCLoss format
                    cost   = criterion(preds1, text, preds_size, length) / batch_size 
            
                else:
                    preds1 = preds1.log_softmax(2).permute(1, 0, 2)
                    cost   = criterion(preds1, text, preds_size, length)           
                
                if opt.CL: 
                    preds2, z2, idx2  = model(view2, text)
                    # compute OCR loss
#                     if opt.baiduCTC:
#                         preds2 = preds2.permute(1, 0, 2)  # to use CTCLoss format
#                         ctc2   = criterion(preds2, text, preds_size, length) / batch_size 
#                         cost   = 0.5*cost + 0.5*ctc2
            
#                     else:
#                         preds2 = preds2.log_softmax(2).permute(1, 0, 2)
#                         ctc2   = criterion(preds2, text, preds_size, length)  
#                         cost   = 0.5*cost + 0.5*ctc2
                    
                    # compute infoNCEloss
                    feature        = torch.cat([z1, z2], dim=0)
                    logits, labels = InfoNCELoss(feature, batch_size)
                    cl_loss        = CELoss(logits, labels)
                    # compute total loss  
                    cost           = cost + lamb*cl_loss

                if opt.multi_task: 
                    # compute recognition blur loss
                    re_loss        = 0.5*(CELoss(idx1, blabel1) + CELoss(idx2, blabel2))
                    # compute total loss  
                    cost           = cost + lamb*re_loss
            
            else:
                # forward
                preds1, z1, idx1 = model(view1, text[:, :-1])  # align with Attention.forward 
                # compute OCR loss
                target = text[:, 1:]  # without [GO] Symbol
                cost   = criterion(preds1.view(-1, preds1.shape[-1]), target.contiguous().view(-1)) 
            
                if opt.CL: 
                    preds2, z2, idx2  = model(view2, text[:, :-1])
                    # compute OCR loss
                    # attn2   = criterion(preds2.view(-1, preds2.shape[-1]), target.contiguous().view(-1)) 
                    # cost    = 0.5*cost + 0.5*attn2
                    
                    # compute info NCE loss
                    feature        = torch.cat([z1, z2], dim=0)
                    logits, labels = InfoNCELoss(feature, batch_size)
                    cl_loss        = CELoss(logits, labels)
                    # compute total loss  
                    cost           = cost + cl_loss/(cl_loss/cost).detach()

                if opt.multi_task:     
                    # compute recognition blur loss
                    re_loss        = 0.5*(CELoss(idx1, blabel1) + CELoss(idx2, blabel2))
                    # compute total loss  
                    cost           = cost + re_loss     
                    
                
            loss_avg.add(cost)
            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
            optimizer.step()
            if opt.cycle_lr:
                scheduler.step()

            if iter % 500 == 0:
                print("iter is "+ str(iter))
                with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
                    log.write(time.strftime("%Y-%m-%d-%H:%M:%S",time.localtime(time.time())) + "at epoch "+ str(epoch)+ " and iter" + str(iter) + ": The training loss is "+str(loss_avg.val())+"\n")

            # validation part
            tmptag = False
            if iteration % opt.valInterval == 0 and not tmptag:
                with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
                    log.write("learning rate is: "+ str(optimizer.state_dict()['param_groups'][0]['lr'])+"\n")
                    model.eval()
                    elapsed_time = time.time() - start_time
                    with torch.no_grad():
                        valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data,new_ed_ac,_,_ = validation(
                        model, criterion, valid_loader, converter, opt)
                    model.train()

                    # training loss and validation loss
                    loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                    loss_avg.reset()

                    current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f},{"new_ed_ac":17s}: {new_ed_ac:0.2f} '

                    # keep best accuracy model (on valid dataset)
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                    if current_norm_ED > best_norm_ED:
                        best_norm_ED = current_norm_ED
                        torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_norm_ED.pth')
                    best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                    loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                    print(loss_model_log)
                    log.write(loss_model_log + '\n')

                    # show some predicted results
                    dashed_line = '-' * 80
                    head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                    predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                    for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                        if 'Attn' in opt.Prediction:
                            gt = gt[:gt.find('[s]')]
                            gt = gt.replace("﻿","")
                            pred = pred[:pred.find('[s]')]

                        predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                    predicted_result_log += f'{dashed_line}'
                    print(predicted_result_log)
                    log.write(predicted_result_log + '\n')
                    
        if epoch % 100 !=0:
            continue
        with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
            log.write("learning rate is: "+ str(optimizer.state_dict()['param_groups'][0]['lr'])+"\n")
            model.eval()
            opt.phase = "test"
            elapsed_time = time.time() - start_time
            with torch.no_grad():
                valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data,new_ed_ac,_,_ = validation(model, criterion, valid_loader, converter, opt)
            model.train()

            # training loss and validation loss
            loss_log = f'[epoch -- {epoch}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
            loss_avg.reset()

            current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f},{"new_ed_ac":17s}: {new_ed_ac:0.2f} '

            # keep best accuracy model (on valid dataset)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
            torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/epoch-{epoch}.pth')
            if current_norm_ED > best_norm_ED:
                best_norm_ED = current_norm_ED
                # torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_norm_ED.pth')
            best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

            loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
            print(loss_model_log)
            log.write(loss_model_log + '\n')

                    # show some predicted results
            dashed_line = '-' * 80
            head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
            predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
            for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                if 'Attn' in opt.Prediction:
                    gt = gt[:gt.find('[s]')]
                    pred = pred[:pred.find('[s]')]

                predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
            predicted_result_log += f'{dashed_line}'
            print(predicted_result_log)
            log.write(predicted_result_log + '\n')



            # save model per 1e+5 iter.
            if (iteration + 1) % 1e+5 == 0:
                torch.save(
                    model.state_dict(), f'./saved_models/{opt.exp_name}/iter_{iteration+1}.pth')

            if (iteration + 1) == opt.num_iter:
                print('end the training')
                sys.exit()
            iteration += 1       
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """ Training config """
    parser.add_argument('--exp_name',    type=str,   default='',     help='Where to store logs and models')
    parser.add_argument('--exp_info',     type=str,   default='training',                               help='description of this experiments')
    parser.add_argument('--train_data',  type=str,   default='/root/autodl-tmp/Databases/train',     help='path to training dataset')
    parser.add_argument('--valid_data',  type=str,   default='/root/autodl-tmp/Databases/eval',              help='path to validation dataset')
    parser.add_argument('--manualSeed',  type=int,   default=2345,   help='for random seed setting')
    parser.add_argument('--workers',     type=int,   default=6,      help='number of data loading workers')
    parser.add_argument('--batch_size',  type=int,   default=200,    help='input batch size')
    parser.add_argument('--max_epoch',   type=int,   default=30,      help='number of iterations to train for')
    parser.add_argument('--num_iter',    type=int,   default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int,   default=2000,   help='Interval between each validation')
    parser.add_argument('--saved_model', type=str,   default='',         help="path to model to continue training")
    parser.add_argument('--FT',          type=bool,  default=False,      help='whether to do fine-tuning')
    parser.add_argument('--adam',        type=bool,  default=True,       help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr',          type=float, default=1,          help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1',       type=float, default=0.9,        help='beta1 for adam. default=0.9')
    parser.add_argument('--rho',         type=float, default=0.95,       help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps',         type=float, default=1e-8,       help='eps for Adadelta. default=1e-8')
    parser.add_argument('--cycle_lr',    type=bool,  default=True,       help='whether using cycle lr shechudle')
    parser.add_argument('--max_lr',      type=float, default=0.002,      help='max learning rate')
    parser.add_argument('--base_lr',     type=float, default=0.00000001, help='base learning rate')
    parser.add_argument('--step_size_up_ratio',type=float, default=0.2,  help='step_size_up_ratio')
    parser.add_argument('--grad_clip',   type=float, default=5,          help='gradient clipping value. default=5')
    parser.add_argument('--use_imgaug',  type=bool,  default=False,      help='use image augment')
    parser.add_argument('--language',    type=str,   default='En',       help='Recognition Language. En|Zh')
    parser.add_argument('--phase',       type=str,   default='train',    help='phase. train|test')

    """ Loss config """
    parser.add_argument('--baiduCTC',    type=bool, default=False,  help='for data_filtering_off mode')
    parser.add_argument('--CL',          type=bool, default=False,   help='use contrastive learning')
    parser.add_argument('--multi_task',  type=bool, default=False,  help='use multi-task learning')

    """ Data processing """
    parser.add_argument('--select_data',            type=str, default='MJ-ST',           help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio',            type=str, default='0.5-0.5',         help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',  help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length',  type=int,  default=25,         help='maximum-label-length')
    parser.add_argument('--imgH',              type=int,  default=32,         help='the height of the input image')
    parser.add_argument('--imgW',              type=int,  default=100,        help='the width of the input image') # for 'EdgeViT', imgW should be 16X, e.g.,96,112,...
    parser.add_argument('--rgb',               type=bool, default=True,       help='use rgb input')
    parser.add_argument('--character',         type=str,  default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--dictfile',          type=str,  default='./keyfile/charset7655.txt',            help='path to dictfile')
    parser.add_argument('--sensitive',         type=bool, default=False,      help='for sensitive character mode')
    parser.add_argument('--PAD',               type=bool, default=True,       help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off',type=bool, default=False,      help='for data_filtering_off mode')

    """ Model Architecture """
    parser.add_argument('--Transformation',    type=str, default='TPS',    help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='RCNN',
                    help='FeatureExtraction stage. RCNN|DenseNet|cnnEdgeViT|SVTR-T|SVTR-S|EdgeViT-XXS|EdgeViT-XS|EfficientFormerV2-S0|EfficientFormerV2-S1|EfficientNet-b0|EfficientNet-b1|EfficientNet-b2|DS-CNN')
    parser.add_argument('--SequenceModeling',  type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction',        type=str, default='CTC',    help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial',      type=int, default=20,       help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel',     type=int, default=1,        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel',    type=int, default=512,      help='the number of output channel of Feature extractor')
    parser.add_argument('--z_channel',         type=int, default=256,      help='the number of dimensions of z')
    parser.add_argument('--hidden_size',       type=int, default=256,      help='the size of the LSTM hidden state')

    opt = parser.parse_args()


    if opt.language == 'Zh':
        list1=[]
        lines = open(opt.dictfile,'r').readlines()
        for line in lines:
            list1.append(line.replace('\n', ''))  
        
        str1=''.join(list1) 
        opt.character=str1
        print(len(opt.character))  #1254

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-CL{opt.CL}-Multitask{opt.multi_task}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """
        
    train(opt)