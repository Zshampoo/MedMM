import math

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
import matplotlib.pyplot as plt
from transformers import AutoModel,BertModel
import torchvision.models as models
from resnet3d import *
device_ = "cuda:0" # Add your device here

class MedMM(nn.Module):
    def __init__(self,
                 image_model_name = 'resnet18',
                 report_model_name = 'chinese-roberta-wwm-ext', # 'bert-base-chinese', 'chinese-roberta-wwm-ext'
                 img_aggregation_type = 'MedMM', # 'MID', 'AVG', 'GA', '3D', 'KDRA', 'CTSA', 'MedMM'
                 input_W = 256, # width of slice
                 input_H = 256, # height of slice
                 input_D = 9, # slice number
                 multimodal_dim = 512,
                 mhsa_heads = 8,
                 dropout_rate = 0.1,
                 mask_columns = 2,
                 bias = False,
                 channels = 3,
                 nb_class = 2,
                 freeze_layers = [0, 1, 2, 3, 4, 5],
                 fc_type = 'img+rep', # 'img', 'rep', 'img+rep'
                 concate_type = 'direct' # 'direct', 'proj'
                 ):

        super(MedMM, self).__init__()

        # init image encoder
        self.Img_model = ImageEncoder(image_model = image_model_name,
                                     aggregation_type = img_aggregation_type,
                                     H = input_H,
                                     W = input_W,
                                     D = input_D,
                                     channels = channels,
                                     mm_dim = multimodal_dim,
                                     num_class = nb_class,
                                     num_heads = mhsa_heads,
                                     bias = bias,
                                     dropout_rate = dropout_rate,
                                     mask_columns = mask_columns,
                                     )

        # init report encoder
        self.Rep_model = RepEncoder(rep_model=report_model_name,freeze_layers=freeze_layers)

        # init fusion and prediction model
        self.Predict_model = Classifier(img_outputdim = self.Img_model._get_img_dim(),
                                        rep_output_dim = 768,
                                        multimodal_dim = multimodal_dim,
                                        bias = bias,
                                        num_class=nb_class,
                                        fc_type=fc_type,
                                        concat_type=concate_type)

    def forward(self, xis, xrs_encoded_inputs, xds_encoded_inputs):
        '''
        xis: input image (batchsize, slice, C, H, W)
        xrs_encoded_inputs: report after tokenizing
        xds_encoded_inputs: KD after tokenizing
        '''

        # Encoding

        ## REP
        xre, _, _ = self.Rep_model(xrs_encoded_inputs)
        xde, _, _ = self.Rep_model(xds_encoded_inputs)

        ## IMG
        xie, slice_scores, region_atts= self.Img_model(xis, xr_region = xre, xr_slice = xde)

        # Interaction
        z = self.Predict_model(xie, xre)
        # z = self.Predict_model(xie, xde)

        return z, slice_scores, region_atts


# Image Encoder
class ImageEncoder(nn.Module):
    def __init__(self,
                 image_model,
                 aggregation_type,
                 H,
                 W,
                 D,
                 channels,
                 mm_dim,
                 num_class,
                 num_heads,
                 bias,
                 dropout_rate,
                 mask_columns):
        super(ImageEncoder, self).__init__()

        # init Resnet
        self.aggregation = aggregation_type
        self.H = H
        self.W = W
        self.slice_num = D
        self.channels = channels
        self.mm_dim = mm_dim
        self.num_class = num_class
        self.num_heads = num_heads
        self.bias = bias
        self.dropout = dropout_rate
        self.mask_columns = mask_columns

        resnet_model, fc_input = self._get_res_basemodel(image_model, aggregation_type, H, W, D, channels)
        if aggregation_type == '3D':
            self.resnet_model_1 = nn.Sequential(*list(resnet_model.module.children())[:-1]) #  drop FC
        else:
            self.resnet_model_1 = nn.Sequential(*list(resnet_model.children())[:-1]) #  drop FC
            self.resnet_model_2 = nn.Sequential(*list(resnet_model.children())[:-2]) #  drop FC and avgpool

        self.fc_input = fc_input

        # GA
        self.gated_img_v = nn.Sequential(
            nn.Linear(self.fc_input, self.mm_dim, bias=self.bias),
            nn.Tanh()
        )
        self.gated_img_u = nn.Sequential(
            nn.Linear(self.fc_input, self.mm_dim, bias=self.bias),
            nn.Sigmoid()
        )
        self.gated_img_w = nn.Linear(mm_dim, 1)

        # MedMM
        # CTSA_COSINE
        self.Proj_REP_cs = nn.Linear(768, self.mm_dim, bias=self.bias)
        self.Proj_SLICE_cs = nn.Linear(self.fc_input, self.mm_dim, bias=self.bias)

        # KDRA_COSINE
        self.proj_KD_q_cs = nn.Linear(self.rep_dim, self.mm_dim, bias=self.bias)
        self.proj_REGION_k_cs = nn.Linear(self.img_dim, self.mm_dim, bias=self.bias)
        # KDRA_MHSA
        self.proj_KD_q_mhsa = nn.Linear(self.rep_dim, self.mm_dim, bias=self.bias)
        self.proj_REGION_k_mhsa = nn.Linear(self.img_dim, self.mm_dim, bias=self.bias)
        self.proj_REGION_v_mhsa = nn.Linear(self.img_dim, self.mm_dim, bias=self.bias)
        self.Proj_OUT_mhsa = nn.Linear(self.mm_dim, self.mm_dim, bias=self.bias)
        self.attention_REGION = Attention(attn_dropout=self.dropout)

    def _get_img_dim(self):
        return self.fc_input

    def _get_res_basemodel(self, image_model, aggregation_type, H, W, D, channels):
        # backbone
        if aggregation_type == '3D':
            model = resnet18(sample_input_W = W,
                             sample_input_H = H,
                             sample_input_D = D,
                             channels = channels,
                             shortcut_type = 'A',
                             no_cuda = False,
                             num_seg_classes=1)
            model = model.to(device_)
        else:
            self.resnet_dict = {"resnet18": models.resnet18(pretrained=True),
                                "resnet34": models.resnet34(pretrained=True),
                                "resnet50": models.resnet50(pretrained=True),
                                "resnet101": models.resnet101(pretrained=True)}
            model = self.resnet_dict[image_model]
        print("Image feature extractor: {0}, aggregation type: {1}", image_model,aggregation_type)
        fc_input = 512
        return model, fc_input

    def transfer_mask(self, mask_list):
        mask_arr = np.full((8, 8), fill_value=1)
        for m in mask_list:
            mask_arr[:, m] = 0
        return mask_arr

    def transpose_qkv(self, X, num_heads):
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1) # reshape ('batch_size', 'slice', 'num_heads', 'num_hiddens' / 'num_heads')
        X = X.permute(0,2,1,3)
        return X

    def reshape_region_withmask(self,z_reg):
        columns = z_reg.shape[-1]
        dim = z_reg.shape[2]

        z_reg_clip = z_reg[:, :, :, :, self.mask_columns[0] : columns - self.mask_columns[1] ] # (batch,slice, 512, 6,8)
        z_reg_clip = z_reg_clip.reshape(z_reg.shape[0]*z_reg.shape[1], z_reg.shape[2], -1) # (batch*slice, 512, 6*8)
        z_reg_clip = z_reg_clip.transpose(-1,-2)# (batch*slice, 6*8, 512)

        return z_reg_clip

    def KDRA_cs(self, batchsize, slicenum, hi, zcls_des):

        # hi (batch,slice, 512, H, W )
        # zcls_des (batch,768)
        zcls_desmm = self.proj_KD_q_cs(zcls_des)  # (batch,mmdim)
        zcls_desmm_expand = zcls_desmm.unsqueeze(dim=1).repeat(1, slicenum, 1).reshape(batchsize * slicenum,-1).squeeze()  # (batch*sclienum,1,mmdim)
        z_reg = self.reshape_region_withmask(hi)
        z_reg_mm = self.proj_REGION_k_cs(z_reg)
        '''cs'''
        zcls_desmm_norm = F.normalize(zcls_desmm_expand, dim=-1).unsqueeze(dim=1)  # (batchssize*slicenum,1,mmdim)
        z_regmm_norm = F.normalize(z_reg_mm, dim=-1)  # (batchssize*slicenum, H*W,mmdim)
        cos_sim = torch.matmul(zcls_desmm_norm, z_regmm_norm.transpose(1, 2))  # (batchssize*slicenum,1,H*W)
        cos_sim_norm = F.softmax(cos_sim, dim=-1)  # (batchssize*slicenum,1, H*W)
        v_weighted = torch.matmul(cos_sim_norm, z_reg)  # (batchssize*slicenum,1, mmdim)
        v_weighted = v_weighted.reshape(batchsize, slicenum, -1)

        return v_weighted, cos_sim_norm

    def KDRA_mhsa(self, batchsize, slicenum, hi, zcls_des):

        # hi (batch,slice, 512, 8,8)
        # zcls_des (batch,768)
        zcls_desmm = self.proj_KD_q_mhsa(zcls_des)  # (batch,mmdim)
        zcls_desmm_expand = zcls_desmm.unsqueeze(dim=1).repeat(1, slicenum, 1).reshape(batchsize*slicenum,-1).unsqueeze(dim=-2)# (batch*sclienum,1,mmdim)
        z_reg = self.reshape_region_withmask(hi)# (batchssize*slicenum, 6*8,512)

        '''MHSA'''
        query = self.transpose_qkv(zcls_desmm_expand,self.num_heads)  # (batchssize*slicenum,num_heads,1,mmdim/num_heads)
        keys = self.transpose_qkv(self.proj_REGION_k_mhsa(z_reg),self.num_heads)  # (batchssize*slicenum,num_heads,H*W,mmdim/num_heads)
        values = self.transpose_qkv(self.proj_REGION_v_mhsa(z_reg),self.num_heads)  # (batchssize*slicenum,num_heads,H*W,mmdim/num_heads)
        outputs, atts = self.attention_REGION(query, keys, values)
        outputs_concat = outputs.squeeze().reshape(batchsize, slicenum, -1)  # (batchssize,slicenum,mmdim)
        region_atts = atts.squeeze()  # (batchssize,slicenum,num_heads,H*W)
        Output = self.Proj_OUT_mhsa(outputs_concat)  # (batchssize,slicenum,mmdim)

        return Output, region_atts

    def CTSA_cs(self,xpool,xdt):
        # xdt (batchssize, 768)
        # xpool (batchssize,slice,512)
        xdmm = self.Proj_REP_cs(xdt) # (batchsize, mmdim)
        xpmm = self.Proj_SLICE_cs(xpool)# (batchsize,slice,mmdim)

        # cosine
        xdmm_norm = F.normalize(xdmm, dim=-1).unsqueeze(dim=1)  # (batch,1,128)
        xpmm_norm = F.normalize(xpmm, dim=-1)  # (batch,11,128)
        cos_sim = torch.matmul(xdmm_norm, xpmm_norm.transpose(1, 2))  # (batch,1,slice)
        cos_sim_norm = F.softmax(cos_sim, dim=-1)  # (batch,1,11)
        v_weighted = torch.matmul(cos_sim_norm, xpool)
        # v_weighted = torch.matmul(cos_sim_norm, xpmm)

        v_weighted = v_weighted.squeeze()  # (batch,512)
        slice_atts = cos_sim_norm.squeeze() # (batch,11)

        return v_weighted,slice_atts

    def GA(self,xpool):
        # xpool (batchssize,slice,512)
        u = self.gated_img_u(xpool)  # (batch, slice, 128)
        v = self.gated_img_v(xpool)  # (batch, slice, 128)
        A = self.gated_img_w(v * u)  # (batch, slice, 1)
        scores = F.softmax(A, dim=1)  # (batch, slice, 1)
        h = torch.matmul(scores.transpose(-2, -1), xpool)
        h = h.squeeze()

        return h,scores

    def forward(self, xis, xr_region = None, xr_slice=None):
        # Encoding
        ## 3D resnet18
        if self.aggregation == '3D':
            h = self.resnet_model_1(xis)
            hi = nn.AdaptiveAvgPool3d((1, 1, 1))(h)
            hi = nn.Flatten()(hi)
            return hi

        ## 2D & 2.5D

        # first squeeze before encoding
        xis = xis.transpose(1, 2)
        batchsize = xis.shape[0]
        xis = xis.reshape(batchsize * self.slice_num, self.channels, self.H, self.W)  #(batch*slice, 1, 256,256)
        hi = self.resnet_model_2(xis)  # hi (batch*slice, 512, 8,8)
        # then expand after encoding
        hi = hi.reshape(batchsize, self.slice_num, 512, 8, 8)  # (batch,slice,512, 8,8)
        h_ = nn.AdaptiveAvgPool2d((1, 1))(hi)  # hi (batch,slice,512, 1, 1)
        h_squeeze = h_.squeeze()  # (batch,slice,512)

        # Aggregation
        v, slice_scores, region_atts = None, None, None
        if self.aggregation == 'MID':
            v = h_squeeze[:,4,:].squeeze() # 4-th slice denotes the middle slice
        elif self.attention_type == 'AVG':
            v = torch.mean(h_squeeze,dim=1) # avg on slice
        elif self.aggregation == 'GA':
            v, slice_scores = self.gated_attention(h_squeeze)
        elif self.aggregation == 'CTSA':
            v, slice_scores = self.CTSA_cs(h_squeeze, xr_slice)
        elif self.aggregation == 'KDRA':
            batchsize = hi.shape[0]
            slicenum = hi.shape[1]
            v_slices, region_atts = self.KDRA_mhsa(batchsize, slicenum, hi, xr_region)
            v = torch.mean(v_slices,dim=1) # avg on slice
        elif self.aggregation == 'MedMM':
            batchsize = hi.shape[0]
            slicenum = hi.shape[1]
            # rsa
            residual, slice_scores = self.CTSA_cs(h_squeeze, xr_slice)
            # kdra
            v_slices, region_atts = self.KDRA_mhsa(batchsize, slicenum, hi, xr_region)
            # slice aggregation
            v = torch.matmul(slice_scores.unsqueeze(dim=1),v_slices).squeeze()
            # residual flow
            v = v + residual
        return v, slice_scores, region_atts

# Report Encoder
class RepEncoder(nn.Module):
    def __init__(self, rep_model, freeze_layers):
        super(RepEncoder, self).__init__()

        # init roberta
        self.roberta_model = self._get_rep_basemodel(rep_model, freeze_layers)

    def _get_rep_basemodel(self, rep_model_name, freeze_layers):
        try:
            print("report feature extractor:", rep_model_name)
            if rep_model_name == 'bert-base-chinese':
                model = AutoModel.from_pretrained(rep_model_name)
            elif rep_model_name == 'chinese-roberta-wwm-ext':
                model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

            print("--------Report pre-train model load successfully --------")
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False

        return model

    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        """
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self,encoded_inputs):
        encoded_inputs = encoded_inputs.to(device_)
        outputs = self.roberta_model(**encoded_inputs)
        mp_embeddings = self.mean_pooling(outputs, encoded_inputs['attention_mask'])
        cls_embeddings = outputs[1]
        token_embeddings = outputs[0]

        return cls_embeddings, mp_embeddings, token_embeddings

# Fusion and Prediction
class Classifier(nn.Module):
    def __init__(self, img_outputdim, rep_output_dim, multimodal_dim, bias, num_class, fc_type = 'multimodal', concat_type = 'direct'):
        super(Classifier, self).__init__()

        self.img_dim = img_outputdim
        self.rep_dim = rep_output_dim
        self.mm_dim = multimodal_dim
        self.bias = bias
        self.num_class = num_class
        self.fc_type = fc_type
        self.concat_type = concat_type

        # PROJECTION MATRICES
        self.proj_img = nn.Linear(self.img_dim,self.mm_dim,bias = self.bias)
        self.proj_rep = nn.Linear(self.rep_dim, self.mm_dim,bias = self.bias)

        # FCs
        ## FC for img_only baselines
        self.FC_img = nn.Sequential(
            nn.Linear(self.img_dim, self.num_class),
            nn.Softmax(dim=-1)
        )
        ## FC for rep_only baselines
        self.FC_rep = nn.Sequential(
            nn.Linear(self.rep_dim, self.num_class),
            nn.Softmax(dim=-1)
        )
        ## FC for multi-modal baselines
        self.FC_mm = nn.Sequential(
            nn.Linear(self.img_dim + self.rep_dim, self.num_class),
            nn.Softmax(dim=-1)
        )

        self.FC_mm_proj = nn.Sequential(
            nn.Linear(self.mm_dim + self.mm_dim, self.num_class),
            nn.Softmax(dim=-1)
        )
        self.MLP_mm_proj = nn.Sequential(
            nn.Linear(self.mm_dim + self.mm_dim, self.mm_dim),
            nn.BatchNorm1d(num_features=self.mm_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.mm_dim, self.num_class),
            nn.Softmax(dim=-1)
        )
    def forward(self, zie, zre):
        z = None

        if self.fc_type == 'img_only':
            z = self.FC_img(zie)
        elif self.fc_type == 'rep_only':
            z = self.FC_rep(zre)
        elif self.fc_type == 'img+rep':
            if self.concat_type == 'direct':
                z_ = torch.cat([zie,zre],dim=-1)
                z = self.FC_mm(z_)
            elif self.concat_type == 'proj':
                zim = self.img_proj(zie)
                zrm = self.rep_proj(zre)
                z_ = torch.cat([zim, zrm], dim=-1)
                z = self.FC_mm_proj(z_)
                # z = self.MLP_mm_proj(z_)
            else:
                print('wrong value of concat_type')
        else:
            print('wrong value of fc_type')

        return z


# Attention Function
class Attention(nn.Module):
    def __init__(self, temperature=None, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        if self.temperature == None:
            self.temperature = math.sqrt(q.shape[-1])
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))
        if mask is not None:
            mask = torch.Tensor(mask).reshape(1,1,1,-1).to(device_)
            mask_ = mask.repeat(attn.shape[0],attn.shape[1],attn.shape[2],1)
            attn  = attn.masked_fill(mask_ == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


