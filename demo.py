import os, time, string, argparse, re,dataset, itertools, math, cv2
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

from torch.nn.functional import cosine_similarity
from torch.autograd import Variable
from imghdr import tests
from operator import mod
from utils import CTCLabelConverter
from model import Model
from PIL import Image
from test_figs import Beam, str_normal, use_language, ctc_bm_decode, context_beam_search


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NEG_INF = float('-inf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """ Testing config """    
    parser.add_argument('--savename',         type=str, default="cnnEdgeViT",      help='the size of the LSTM hidden state')
    parser.add_argument('--imgpath',          type=str, default="./imgs/1012.png",      help='the size of the LSTM hidden state')
    parser.add_argument('--batch_size',       type=int,  default=1,    help='maximum-label-length')
    parser.add_argument('--saved_model',      type=str,  default="./saved_models/None-cnnEdgeViT-None-CTC/best_accuracy.pth", 
                        help='model checkpoint file')
    parser.add_argument('--batch_max_length', type=int,  default=150,   help='maximum-label-length')
    parser.add_argument('--imgH',             type=int,  default=32,   help='the height of the input image')
    parser.add_argument('--imgW',             type=int,  default=1600,  help='the width of the input image')
    parser.add_argument('--rgb',              type=bool, default=True, help='use rgb input')
    parser.add_argument('--language',         type=str,  default='Zh',       help='Recognition Language. En|Zh')
    parser.add_argument('--phase',            type=str,  default='test',     help='phase')
    parser.add_argument('--CL',               type=bool, default=False,  help='use contrastive learning')
    parser.add_argument('--multi_task',       type=bool, default=False,  help='use multi-task learning')
    parser.add_argument('--uselm',            type=str,  default=False)    
    """ Data processing """
    parser.add_argument('--character', type=str, default='', help='character label')
    parser.add_argument('--dictfile',default='./keyfile/charset7655.txt')
    parser.add_argument('--PAD',default=True, action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation',   type=str,  default="None")
    parser.add_argument('--num_fiducial',     type=int,  default=20,       help='number of fiducial points of TPS-STN')
    parser.add_argument('--FeatureExtraction',type=str,  default="cnnEdgeViT")
    parser.add_argument('--SequenceModeling', type=str,  default="None", help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction',       type=str,  default="CTC")
    parser.add_argument('--input_channel',    type=int,  default=3)
    parser.add_argument('--output_channel',   type=int,  default=256,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--z_channel',         type=int, default=192,      help='the number of dimensions of z')
    parser.add_argument('--hidden_size',       type=int, default=256,      help='the size of the LSTM hidden state')

    opt = parser.parse_args([])
    
    if opt.language == 'Zh':
        list1=[]
        lines = open(opt.dictfile,'r').readlines()
        for line in lines:
            list1.append(line.replace('\n', ''))  
        
        char_set = ['[CTCblank]'] + list1 
        str1=''.join(list1) 
        opt.character=str1
        print(len(opt.character))  

    cudnn.benchmark = True
    cudnn.deterministic = True

    converter = CTCLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    print(model)
    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    model.eval()

    img = Image.open(opt.imgpath).convert("RGB")
    w, h = img.size
    ratio = w / float(h)
    resized_w = math.ceil(32 * ratio)
    padsize = opt.imgW
    if resized_w > opt.imgW:
        resized_w = opt.imgW

    with torch.no_grad():
        transformer = dataset.NormalizePAD((3,32,padsize))
        resized_image = img.resize((resized_w, 32), Image.BICUBIC)
        img_tensor = transformer(resized_image)        
        img_tensor = img_tensor.view(1,*img_tensor.size()).cuda()
                        
        if 'CTC' in opt.Prediction:
            text_for_pred = ""
            preds, _, _ = model(img_tensor, text_for_pred)
            pred_lm = ""
            if opt.uselm:
                _,preds_lm = ctc_bm_decode(preds=preds)
                pred_lm = preds_lm[0]
            preds_size = torch.IntTensor([preds.size(1)] * opt.batch_size)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index.data, preds_size.data)[0]

    print('识别结果: ', preds_str)