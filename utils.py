import torch, random, os, yaml
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from torch import Tensor
from straug.blur import GaussianBlur, DefocusBlur, MotionBlur

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts


class CTCLabelConverterForBaiduWarpctc(object):
    """ Convert between text-label and text-index for baidu warpctc """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

     
class augment_dec(object):
    def __init__(self, device, in_channels=3):
        Gaussian = GaussianBlur()
        Defocus  = DefocusBlur(device=device, nIn=in_channels)
        Motion   = MotionBlur(device=device, nIn=in_channels)
        # Zoom     = ZoomBlur()
        # Identity = nn.Identity()
        
        self.blur_dict = [Gaussian, Defocus, Motion] 
    
    def __call__(self, minibatch, idx=None):
        '''
         mag(int): blur level
        '''
        if idx is not None:
            blur  = self.blur_dict[idx]
        else:
            idx   = random.randint(0, int(len(self.blur_dict)-1))
            blur  = self.blur_dict[idx]
                             
        blurred   = blur(minibatch)
        return blurred,  torch.tensor([idx+1]) # idx=1~4, w/o blur idx=0
    
    
def FL_InfoNCELoss(features, batch_size, n_views=2, temperature=0.07):
    '''
     features: [org, augement]
    '''
    # labels = [1,2,3,..... b, 1,2,3,....b] repreat n views times
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    # generate diagonal labels; where 
    # "labels = labels.unsqueeze(0) == labels.unsqueeze(1)" 
    # mean: if "labels.unsqueeze(0)[idx]" = ele in "labels.unsqueeze(1)[idx]"
    # labels[idx] = 1, else labels[idx] = 0
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() # 64 x 64 diagonal labels
    labels = labels.cuda()
  
    '''
    calculate 1d similarity
    '''
    # features = F.normalize(features, dim=1) # (b, f)
    # similarity_matrix = torch.matmul(features, features.T)
    '''
    calculate 2d similarity
    '''
    features = F.normalize(features, dim=(-2, -1)) # (n_view*b, c, f)
    # similarity_matrix = torch.einsum("mjk,nkj->mn", [features, features.permute(0,2,1)])
    
    sim_list = []
    for sample in features:
        # sample.shape = [class, frames]
        # 计算每个sample之间的相似度
        sim = cosine_similarity(sample, features, dim=1).mean(-1)
        sim_list.append(sim)  
        
    similarity_matrix = torch.stack(sim_list, dim=0).contiguous()
    
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    # mask = np.bool(labels)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1) # ~: if mask is int tensor, ~mask = mask-1; if mask is bool tensor, ~mask = reverse mask
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    logits = logits / temperature
    return logits, labels



def InfoNCELoss(features, batch_size, n_views=2, temperature=0.07):
    '''
     features: [org, augement]
    '''
    # labels = [1,2,3,..... b, 1,2,3,....b] repreat n views times
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    # generate diagonal labels; where 
    # "labels = labels.unsqueeze(0) == labels.unsqueeze(1)" 
    # mean: if "labels.unsqueeze(0)[idx]" = ele in "labels.unsqueeze(1)[idx]"
    # labels[idx] = 1, else labels[idx] = 0
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() # 64 x 64 diagonal labels
    labels = labels.cuda()
  
    '''
    calculate 1d similarity
    '''
    features = F.normalize(features, dim=1) # (b, f)
    similarity_matrix = torch.matmul(features, features.T)
    
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    # mask = np.bool(labels)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1) # ~: if mask is int tensor, ~mask = mask-1; if mask is bool tensor, ~mask = reverse mask
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    logits = logits / temperature
    return logits, labels