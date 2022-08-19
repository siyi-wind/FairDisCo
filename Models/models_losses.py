'''
Class Network is used to construct the model architecture
Class Confusion_Loss
Class Supervised_Contrastive_Loss
'''

import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class Network(torch.nn.Module):
    def __init__(self, choice='vgg16', output_size=9, pretrained=True) :
        '''
        output_size: int  only one output
                     list  first is skin type, second is skin conditipon (used in disentangle, attribute_aware)
        '''
        super(Network, self).__init__()
        self.choice = choice
        bottle_neck = 256

        if self.choice == 'vgg16':
            self.feature_extractor = models.vgg16(pretrained=pretrained)
            num_ftrs = self.feature_extractor.classifier[6].in_features
            self.feature_extractor.classifier[6] = nn.Linear(num_ftrs, output_size)
        
        if self.choice == 'resnet18':
            self.feature_extractor = models.resnet18(pretrained=pretrained)
            num_ftrs = self.feature_extractor.fc.in_features
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
            self.classifier = nn.Linear(num_ftrs, output_size)
            self.project_head = nn.Sequential(
                 nn.Linear(num_ftrs, 512),
                 nn.BatchNorm1d(512),
                 nn.ReLU(inplace=True),
                 nn.Linear(512, 128),
            )
            
        if self.choice == 'disentangle':
            self.feature_extractor = models.resnet18(pretrained=pretrained)
            num_ftrs = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Linear(num_ftrs, bottle_neck)
            # for contrastive loss
            self.project_head = nn.Sequential(
                 nn.Linear(bottle_neck, 512),
                 nn.BatchNorm1d(512),
                 nn.ReLU(inplace=True),
                 nn.Linear(512, 128),
            )
            # self.activation = torch.nn.ReLU()
            # branch 1
            self.branch_1 = nn.Linear(bottle_neck, output_size[0])
            # branch 2
            self.branch_2 = nn.Linear(bottle_neck, output_size[1])
        
        if self.choice == 'attribute_aware':
            # use sensitive information into the network to train
            bottle_neck = 256
            self.feature_extractor = models.resnet18(pretrained=pretrained)
            num_ftrs = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Linear(num_ftrs, bottle_neck)
            self.attribute_layer = nn.Linear(output_size[1], bottle_neck) 
            self.classifier = nn.Linear(bottle_neck, output_size[0])

    
    def forward(self, x, attribute=None):
        if self.choice == 'disentangle':
            feature_map = self.feature_extractor(x)  # (bs, bottle_neck)
            out_1 = self.branch_1(feature_map)
            out_2 = self.branch_2(feature_map)
            out_4 = self.project_head(feature_map)
            # detach feature map and pass though branch 2 again
            feature_map_detach = feature_map.detach()
            out_3 = self.branch_2(feature_map_detach)
            return [out_1, out_2, out_3, out_4]
            # return [out_1, out_2, out_3]
            
        elif self.choice == 'attribute_aware':
            feature_map = self.feature_extractor(x) # (bs, bottle_neck)
            attribute_upsample = self.attribute_layer(attribute) # (bs, bottle_neck)
            fused_feature = feature_map+attribute_upsample # (bs, bottle_neck)
            fused_feature = F.relu(fused_feature) # (bs, bottle_neck)
            out = self.classifier(fused_feature)
            return out

        else:
            output = self.feature_extractor(x)
            output = output.view(x.size(0), -1)
            out1 = self.classifier(output)
            out2 = self.project_head(output)
            return [out1, out2]


class Confusion_Loss(torch.nn.Module):
    '''
    Confusion loss built based on the paper 'Invesgating bias and fairness.....' 
    (https://www.repository.cam.ac.uk/bitstream/handle/1810/309834/XuEtAl-ECCV2020W.pdf?sequence=1&isAllowed=y)
    '''
    def __init__(self):
        super(Confusion_Loss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, output, label):
        # output (bs, out_size). label (bs)
        prediction = self.softmax(output) # (bs, out_size)
        log_prediction = torch.log(prediction)
        loss = -torch.mean(torch.mean(log_prediction, dim=1), dim=0)

        # loss = torch.mean(torch.mean(prediction*log_prediction, dim=1), dim=0)
        return loss


class Supervised_Contrastive_Loss(torch.nn.Module):
    '''
    from https://github.com/GuillaumeErhard/Supervised_contrastive_loss_pytorch/blob/main/loss/spc.py
    https://blog.csdn.net/wf19971210/article/details/116715880
    Treat samples in the same labels as the positive samples, others as negative samples
    '''
    def __init__(self, temperature=0.1, device='cpu'):
        super(Supervised_Contrastive_Loss, self).__init__()
        self.temperature = temperature
        self.device = device
    
    def forward(self, projections, targets, attribute=None):
        # projections (bs, dim), targets (bs)
        # similarity matrix/T
        dot_product_tempered = F.cosine_similarity(projections.unsqueeze(1), projections.unsqueeze(0),dim=2)/self.temperature
        # print(dot_product_tempered)
        exp_dot_tempered = torch.exp(dot_product_tempered- torch.max(dot_product_tempered, dim=1, keepdim=True)[0])+ 1e-5
        # a matrix, same labels are true, others are false
        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(self.device)
        # a matrix, diagonal are zeros, others are ones
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(self.device)
        mask_nonsimilar_class = ~mask_similar_class
        # a matrix, same labels are 1, others are 0, and diagonal are zeros
        mask_combined = mask_similar_class * mask_anchor_out
        # num of similar samples for sample
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        # print(exp_dot_tempered * mask_nonsimilar_class* mask_similar_attr)
        # print(torch.sum(exp_dot_tempered * mask_nonsimilar_class* mask_similar_attr, dim=1, keepdim=True)+exp_dot_tempered)
        if attribute != None:
            mask_similar_attr = (attribute.unsqueeze(1).repeat(1, attribute.shape[0]) == attribute).to(self.device)
            log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_nonsimilar_class * mask_similar_attr, dim=1, keepdim=True)+exp_dot_tempered+1e-5))
       
        else:
            log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_nonsimilar_class, dim=1, keepdim=True)+exp_dot_tempered+1e-5))
        supervised_contrastive_loss = torch.sum(log_prob * mask_combined)/(torch.sum(cardinality_per_samples)+1e-5)

        
        return supervised_contrastive_loss

        
        




if __name__ == '__main__':
    pass
# check network
    # n_epochs = 50
    # level = 'high'
    # holdout_set = 'random_holdout'
    # model = Sensitive_Network('orthogonal', output_size=6, encoder_link='models_saved/32/model_path_{}_{}_{}.pt'.format(n_epochs,level,holdout_set)).cuda()
    # x = torch.randn(64, 3, 224, 224).cuda()
    # y = model(x)
    # print(y.shape)
    # print(model.state_dict().keys())


    # check supervised contrastive loss
    # loss_func = Supervised_Contrastive_Loss()
    # a,b = torch.tensor([[0.,0,0,0,1,1,1,1,1,1]]), torch.tensor([[1.,1,1,1,0,0,0,0,0,0]])
    # a,b  = torch.ones((3,7)), torch.ones(3,7)
    # a,b = a.repeat((3,1)), b.repeat((3,1))
    # a = torch.tensor([[0.,0,1,1]])
    # a= a.repeat((6,1))
    # a = torch.randn(3,10)
    # b = torch.tensor([[1.,1,1,1,0,0,0,0,0,0]])
    # c,d = torch.ones(3), torch.zeros(1)
    # x = torch.cat((a,b),dim=0)

    # y = torch.tensor([1,1,1,0,0,0])
    # z = torch.tensor([2,3,3,2,3,3])
    # loss = loss_func(x, y, z)
    # print(loss)
