import copy
import logging
import torch
from torch import nn
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear
import timm
import torch.nn.functional as F
from convs.projections import Proj_Pure_MLP, MultiHeadAttention
from utils.toolkit import get_attribute
from convs.adapter import AdapterLayer
from convs.lora import LoRALinear

def get_convnet(args, pretrained=False):

    backbone_name = args["convnet_type"].lower()
    algorithm_name = args["model_name"].lower()
    if 'clip' in backbone_name:
        print('Using CLIP model as the backbone')
        import open_clip
        if backbone_name == 'clip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        elif backbone_name=='clip_laion2b':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        elif backbone_name=='openai_clip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        else:
            raise NotImplementedError("Unknown type {}".format(backbone_name))
    
    else:
        raise NotImplementedError("Unknown type {}".format(backbone_name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)
        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self


class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )



class CosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )

        return fc


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = (
            self.alpha * x[:, low_range:high_range] + self.beta
        )
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
    def __init__(self, args, pretrained, bias_correction=False):
        super().__init__(args, pretrained)

        # Bias layer
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        if self.bias_correction:
            logits = out["logits"]
            for i, layer in enumerate(self.bias_layers):
                logits = layer(
                    logits, sum(self.task_sizes[:i]), sum(self.task_sizes[: i + 1])
                )
            out["logits"] = logits

        out.update(x)

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True



class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.convnet, self.preprocess, self.tokenizer = get_convnet(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet.encode_image(x)

    def encode_image(self, x):
        return self.convnet.encode_image(x)
    
    def encode_text(self, x):
        return self.convnet.encode_text(x)
        
    def forward(self, x):
        x = self.convnet.encode_image(x)
        out = self.fc(x)
        return out



class SimpleClipNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

        self.convnet, self.preprocess, self.tokenizer = get_convnet(args, pretrained)
        self.class_name = 'SimpleClipNet'
        self.args = args


    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet.encode_image(x)

    def encode_image(self, x):
        return self.convnet.encode_image(x)
    
    def encode_text(self, x):
        return self.convnet.encode_text(x)

    def forward(self, img, text):

        image_features, text_features, logit_scale=self.convnet(img, text)
        return image_features, text_features, logit_scale

    def re_initiate(self):
        print('re-initiate model')
        self.convnet, self.preprocess, self.tokenizer = get_convnet(self.args, True)


class Proof_Net(SimpleClipNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.projs_img = nn.ModuleList()
        self.projs_text = nn.ModuleList()
        self.args = args
        self._device = args["device"][0]
        self.projtype = get_attribute(self.args, 'projection_type', 'mlp')
        self.context_prompt_length_per_task = get_attribute(self.args, 'context_prompt_length_per_task', 3)
        
        self.sel_attn = MultiHeadAttention(1, self.feature_dim, self.feature_dim, self.feature_dim, dropout=0.1)
        self.img_prototypes = None

        self.context_prompts = nn.ParameterList()

    def update_prototype(self, nb_classes):
        if self.img_prototypes is not None:
            nb_output = len(self.img_prototypes)
            self.img_prototypes = torch.cat([copy.deepcopy(self.img_prototypes).to(self._device), torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)]).to(self._device)
        else:
            self.img_prototypes = torch.zeros(nb_classes, self.feature_dim).to(self._device)
        print('update prototype, now we have {} prototypes'.format(self.img_prototypes.shape[0]))
    
    def update_context_prompt(self):
        for i in range(len(self.context_prompts)):
            self.context_prompts[i].requires_grad = False
        self.context_prompts.append(nn.Parameter(torch.randn(self.context_prompt_length_per_task, self.feature_dim).to(self._device)))
        print('update context prompt, now we have {} context prompts'.format(len(self.context_prompts) * self.context_prompt_length_per_task))
        self.context_prompts.to(self._device)
    
    def get_context_prompts(self):
        return torch.cat([item for item in self.context_prompts], dim=0)

    def encode_image(self, x, normalize: bool = False):
        x = x.to(self._device)
        basic_img_features = self.convnet.encode_image(x)
        img_features = [proj(basic_img_features) for proj in self.projs_img]
        img_features = torch.stack(img_features, dim=1)#[bs,num_proj,dim]
        image_feas = torch.sum(img_features, dim=1)#[bs,dim]
        return F.normalize(image_feas, dim=-1) if normalize else image_feas
        
    def encode_text(self, x, normalize: bool = False):
        x = x.to(self._device)
        basic_text_features = self.convnet.encode_text(x)
        text_features = [proj(basic_text_features) for proj in self.projs_text]
        text_features = torch.stack(text_features, dim=1)
        text_feas = torch.sum(text_features, dim=1) #[bs,dim]
        return F.normalize(text_feas, dim=-1) if normalize else text_feas
        
    def encode_prototpyes(self, normalize: bool = False):
        self.img_prototypes=self.img_prototypes.to(self._device)
        img_features = [proj(self.img_prototypes) for proj in self.projs_img]
        img_features=torch.stack(img_features, dim=1)#[nb_class,num_proj,dim]
        image_feas=torch.sum(img_features, dim=1)#[nb_class,dim]
        return F.normalize(image_feas, dim=-1) if normalize else image_feas

    def extend_task(self):
        self.projs_img.append(self.extend_item())
        self.projs_text.append(self.extend_item())

    def extend_item(self):
        if self.projtype=='pure_mlp':
            return Proj_Pure_MLP(self.feature_dim,self.feature_dim,self.feature_dim).to(self._device)
        else:
            raise NotImplementedError
    
    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)#bs,dim
        text_features = self.encode_text(text, normalize=True)#bs,dim

        prototype_features = self.encode_prototpyes(normalize=True) #nb_class,dim
        context_prompts=self.get_context_prompts() # num_prompt, dim

        len_texts=text_features.shape[0]
        len_protos=prototype_features.shape[0]
        len_context_prompts=context_prompts.shape[0]
        # restack the features and pass them through the attention layer
        image_features = image_features.view(image_features.shape[0], -1, self.feature_dim)#bs,1,dim
        text_features = text_features.view(text_features.shape[0], self.feature_dim)#num_text,dim
        prototype_features = prototype_features.view(prototype_features.shape[0], self.feature_dim)#len_proto,dim
        context_prompts = context_prompts.view(context_prompts.shape[0], self.feature_dim)#len_con,dim
        # expand text features to be the same dim as image features
        text_features = text_features.expand(image_features.shape[0], text_features.shape[0], self.feature_dim)#bs,num_text,dim
        prototype_features = prototype_features.expand(image_features.shape[0], prototype_features.shape[0], self.feature_dim)#bs,len_proto,dim
        context_prompts = context_prompts.expand(image_features.shape[0], context_prompts.shape[0], self.feature_dim)#bs,len_con,dim
        # concat them together
        # features = torch.cat([image_features, text_features, prototype_features], dim=1) # bsize * (1+num_texts+num_protos) * dim
        features = torch.cat([image_features, text_features, prototype_features, context_prompts], dim=1) # bsize * (1+num_texts+num_protos+num_context) * dim
        # pass through the attention layer
        features = self.sel_attn(features, features, features)
        # split them back, image features are the first half, text features are the second half
        # image_features, text_features = torch.split(features, features.shape[1] // 2, dim=1)
        image_features = features[:, 0, :] # bsize * dim
        text_features = features[:, 1:len_texts+1, :] # bsize * num_texts * dim
        prototype_features = features[:, len_texts+1:len_texts+1+len_protos, :] # bsize * num_protos * dim 
        context_prompts = features[:, len_texts+1+len_protos:len_texts+1+len_protos+len_context_prompts, :] # bsize * num_context * dim
        # remove the 0-th dimension of text features to be num_texts * dim
        text_features = torch.mean(text_features, dim=0) # num_texts * dim
        prototype_features = torch.mean(prototype_features, dim=0) # num_protos * dim
        # squeeze
        image_features = image_features.view(image_features.shape[0], -1)
        text_features = text_features.view(text_features.shape[0], -1)
        prototype_features = prototype_features.view(prototype_features.shape[0], -1)
        return image_features, text_features, self.convnet.logit_scale.exp(), prototype_features
    
    def forward_transformer(self, image_features, text_features, transformer=False):
        prototype_features = self.encode_prototpyes(normalize=True)
        if transformer:
            context_prompts = self.get_context_prompts()
            len_texts = text_features.shape[0]
            len_protos = prototype_features.shape[0]
            len_context_prompts = context_prompts.shape[0]
            # restack the features and pass them through the attention layer
            image_features = image_features.view(image_features.shape[0], -1, self.feature_dim) #[bs, 1, dim]
            text_features = text_features.view(text_features.shape[0], self.feature_dim) #[total_classes, dim]
            prototype_features = prototype_features.view(prototype_features.shape[0], self.feature_dim) #[len_pro, dim]
            context_prompts = context_prompts.view(context_prompts.shape[0], self.feature_dim) #[len_con_pro, dim]
            # expand text features to be the same dim as image features
            text_features = text_features.expand(image_features.shape[0], text_features.shape[0], self.feature_dim) #[bs, total_classes, dim]
            prototype_features = prototype_features.expand(image_features.shape[0], prototype_features.shape[0], self.feature_dim) #[bs, len_pro, dim]
            context_prompts = context_prompts.expand(image_features.shape[0], context_prompts.shape[0], self.feature_dim) #[bs, len_con_pro, dim]
            # concat them together
            # features = torch.cat([image_features, text_features, prototype_features], dim=1) # bsize * (1+num_texts+num_protos) * dim
            features = torch.cat([image_features, text_features, prototype_features, context_prompts], dim=1) # bsize * (1+num_texts+num_protos+num_context) * dim
            # pass through the attention layer
            features = self.sel_attn(features, features, features)
            # split them back, image features are the first half, text features are the second half
            # image_features, text_features = torch.split(features, features.shape[1] // 2, dim=1)
            image_features = features[:, 0, :] # bsize * dim
            text_features = features[:, 1:len_texts+1, :] # bsize * num_texts * dim
            prototype_features = features[:, len_texts+1:len_texts+1+len_protos, :] # bsize * num_protos * dim 
            context_prompts = features[:, len_texts+1+len_protos:len_texts+1+len_protos+len_context_prompts, :] # bsize * num_context * dim
            # remove the 0-th dimension of text features to be num_texts * dim
            text_features = torch.mean(text_features, dim=0) # num_texts * dim
            prototype_features = torch.mean(prototype_features, dim=0) # num_protos * dim
            # squeeze
            image_features = image_features.view(image_features.shape[0], -1)
            text_features = text_features.view(text_features.shape[0], -1)
            prototype_features = prototype_features.view(prototype_features.shape[0], -1)
            return image_features, text_features, self.convnet.logit_scale.exp(), prototype_features
        else:
            return image_features, text_features, self.convnet.logit_scale.exp(), prototype_features
    
    
    def freeze_projection_weight_new(self):
        if len(self.projs_img)>1:
            for i in range(len(self.projs_img)):
                for param in self.projs_img[i].parameters():
                    param.requires_grad = False
                for param in self.projs_text[i].parameters():
                    param.requires_grad = True
            for param in self.projs_img[-1].parameters():
                param.requires_grad = True
        for param in self.sel_attn.parameters():
            param.requires_grad = True

class PeftClipNet(SimpleClipNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.args = args
        self._device = self.args["device"][0]

    def encode_image(self, x, normalize: bool = False):
        x = x.to(self._device)
        image_feas=self.convnet.encode_image(x)
        return F.normalize(image_feas, dim=-1) if normalize else image_feas
        
    def encode_text(self, x, normalize: bool = False):
        x = x.to(self._device)
        text_feas=self.convnet.encode_text(x)
        return F.normalize(text_feas, dim=-1) if normalize else text_feas
     
    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)#bs,dim
        text_features = self.encode_text(text, normalize=True)#bs,dim
        return image_features, text_features, self.convnet.logit_scale.exp()
    
#注意，这里我考虑了两种adapter，一种是加在proj后面的，另一种就是在CLIP内部的Transformer块后面加
class AdapterClipNet(PeftClipNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.num_adapters = get_attribute(self.args, 'num_adapters', 1)
        self.adapter_dim = get_attribute(self.args, 'adapter_dim', 64)
        self.residual_scale = get_attribute(self.args, 'residual_scale', 0.6)  # 残差连接比例
        
        # 存储Adapter模块
        self.adapters = nn.ModuleList()
        
        # 初始化Adapter

        self._add_adapter(is_visual=True)
        self._add_adapter(is_visual=False)
        self.visual_proj_adapter = AdapterLayer(512,adapter_dim=self.adapter_dim,alpha=self.residual_scale)
        self.adapters.append(self.visual_proj_adapter)
        self.textual_proj_adapter = AdapterLayer(512,adapter_dim=self.adapter_dim,alpha=self.residual_scale)
        self.adapters.append(self.textual_proj_adapter)

        #冻结
        self.freeze_backbone()
    
    def _add_adapter(self,is_visual=False):
        blocks= self.convnet.visual.transformer.resblocks if is_visual else self.convnet.transformer.resblocks
        num_layers=len(blocks)

        # 计算要添加Adapter的层索引（从后往前）
        start_idx = max(0, num_layers - self.num_adapters+1)#这里把proj的adapter给减掉
        adapter_indices = list(range(start_idx, num_layers))

        # 为选中的层添加Adapter
        for idx in adapter_indices:
            layer = blocks[idx]
            adapter = AdapterLayer(768,adapter_dim=self.adapter_dim,alpha=self.residual_scale) if is_visual else AdapterLayer(512,adapter_dim=self.adapter_dim,alpha=self.residual_scale)
            self.adapters.append(adapter)
            # 使用hook机制插入Adapter
            self._add_adapter_to_layer(layer, adapter)

    # 只针对transformer层的adapter，不是最后投影层的
    def _add_adapter_to_layer(self, layer, adapter):
        def adapter_hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            
            # 维度转换：CLIP Transformer默认返回 (seq_len, batch_size, dim)，转为 (batch_size, seq_len, dim)
            out_bsz = out.permute(1, 0, 2)  # (batch, seq, dim)
            
            # 1. 提取CLS token（仅对第0位token应用Adapter）
            cls_token = out_bsz[:, 0, :]  # (batch, dim)
            
            # 2. 通过Adapter处理CLS token
            adapted_cls = adapter(cls_token)  # (batch, dim)
            
            # 3. 安全替换CLS token
            new_out_bsz = out_bsz.clone()  #这里要小心修改计算图的问题
            new_out_bsz[:, 0, :] = adapted_cls  # 仅替换CLS token，保留其他token的原始特征
            
            # 4. 恢复原始维度顺序 (seq_len, batch_size, dim)
            new_out = new_out_bsz.permute(1, 0, 2)
            
            # 5. 保持输出格式与原模块一致（输入是tuple）
            if isinstance(output, tuple):
                return (new_out,) + output[1:]
            return new_out
        
        # 注册forward hook，注入Adapter逻辑
        layer.register_forward_hook(adapter_hook)

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)#bs,dim
        text_features = self.encode_text(text, normalize=True)#bs,dim
        image_features=self.visual_proj_adapter(image_features)
        text_features=self.textual_proj_adapter(text_features)
        return image_features, text_features, self.convnet.logit_scale.exp()

    def freeze_backbone(self):
        # 冻结整个CLIP模型
        for param in self.convnet.parameters():
            param.requires_grad = False
        # 只保持Adapter可训练
        for adapter in self.adapters:
            for param in adapter.parameters():
                param.requires_grad = True

class LoRAClipNet(PeftClipNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.lora_rank =get_attribute(args, 'lora_rank', 8)
        self.lora_alpha = get_attribute(args, 'lora_alpha', 16.0)
        self.num_lora=get_attribute(args,"num_lora",1)
        
        self.lora_modules = nn.ModuleDict()

        self._add_lora(is_visual=True)
        self._add_lora(is_visual=False)

        self._freeze_backbone()
    
    def _add_lora(self,is_visual=True):
        blocks=self.convnet.visual.transformer.resblocks if is_visual==True else self.convnet.visual.transformer.resblocks
        num_layers=len(blocks)
        start_idx= max(0,  num_layers - self.num_lora)
        lora_indices=list(range(start_idx,  num_layers))
        for idx in lora_indices:
            attn =blocks[idx].attn
            self._replace_linear_with_lora(attn, f'visual_attn_{idx}')

    def _replace_linear_with_lora(self, attn_module, module_name):
        # 替换qkv投影层，这里主要是MHA的实现上把三个合在一起投影了，所以就暂时没拆分
        if hasattr(attn_module, 'in_proj'):
            orig_linear = attn_module.in_proj
            lora_linear = LoRALinear(orig_linear, self.lora_rank, self.lora_alpha)
            attn_module.in_proj = lora_linear
            self.lora_modules[f'{module_name}_in_proj'] = lora_linear
        
        # 替换输出投影层
        if hasattr(attn_module, 'out_proj'):
            orig_linear = attn_module.out_proj
            lora_linear = LoRALinear(orig_linear, self.lora_rank, self.lora_alpha)
            attn_module.out_proj = lora_linear
            self.lora_modules[f'{module_name}_out_proj'] = lora_linear
    

    def _freeze_backbone(self):
        for param in self.convnet.parameters():
            param.requires_grad = False
        
        # 确保 LoRA 参数是可训练的
        for module in self.lora_modules.values():
            for param in module.parameters():
                param.requires_grad = True

class PromptClipNet(PeftClipNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.prompt_length = get_attribute(args, 'prompt_length',40)
        self.prompt_layers = get_attribute(args, 'prompt_layers', 8)
        self.prompt_init = get_attribute(args, 'prompt_init', 'random')
        
        # 只在视觉侧添加prompt
        self.visual_prompts = nn.ParameterList()
        self._visual_hooks = []
        
        # 获取视觉编码器维度
        self.visual_dim = 768  # ViT-B/16适用，主要是我没找到正确的接口，目前先硬编码吧

        self.original_seq_len=197
        # 初始化prompt
        self._setup_visual_prompts()
        
        # 冻结主干网络
        self._freeze_backbone()

    def _setup_visual_prompts(self):
        visual_encoder = self.convnet.visual
        num_layers = len(visual_encoder.transformer.resblocks)
        
        # 选择要添加prompt的层（最后几层）
        start_idx = max(0, num_layers - self.prompt_layers)
        prompt_indices = list(range(start_idx, num_layers))
        
        for idx in prompt_indices:
            prompt = self._init_prompt_parameter(self.visual_dim)
            self.visual_prompts.append(prompt)
            
            layer = visual_encoder.transformer.resblocks[idx]
            hook = self._add_prompt_to_visual_layer(layer, prompt, idx)
            self._visual_hooks.append(hook)

    def _init_prompt_parameter(self, dim):
        """初始化prompt参数"""
        if self.prompt_init == 'zeros':
            prompt = torch.zeros(self.prompt_length, dim)#prompt由prompt_length个可学习的token组成
        elif self.prompt_init == 'uniform':
            prompt = torch.empty(self.prompt_length, dim)
            nn.init.uniform_(prompt, -0.1, 0.1)
        else:  # random
            prompt = torch.randn(self.prompt_length, dim) * 0.02
            
        return nn.Parameter(prompt.to(self._device))

    def _add_prompt_to_visual_layer(self, layer, prompt, layer_idx):
        def prompt_pre_hook(module, input):
            # 输入: (seq_len, batch_size, dim)
            # 注意vision-encoder通常没有attention mask，所以比较容易一些，我之前尝试在text-encoder加，就会有这个mask的问题
            x = input[0]
            batch_size = x.shape[1]

            # 检查当前序列长度，确保不会累积
            current_seq_len = x.shape[0]
            if current_seq_len > self.original_seq_len:
                # 如果序列长度已经增加（可能因为前一层的prompt），恢复原始序列
                # 保留CLS token和原始patch tokens
                x = torch.cat([x[:1], x[1+self.prompt_length:]], dim=0)
                # 现在x的形状应该是 (original_seq_len, batch_size, dim)

            # 扩展prompt到batch size
            prompt_expanded = prompt.unsqueeze(1).expand(-1, batch_size, -1)
            
            # 将prompt添加到CLS token之后
            # 原始序列: [CLS, patch1, patch2, ...]
            # 新序列: [CLS, prompt1, prompt2, ..., patch1, patch2, ...]
            new_x = torch.cat([x[:1, :, :], prompt_expanded, x[1:, :, :]], dim=0)
            #这里写的时候容易出错，具体来说x[:1, :, :]不要写成x[0, :, :]，因为后者少了一个维度
            #print("the new sequence length is",len(new_x))
            # 返回修改后的输入
            if len(input) > 1:
                return (new_x,) + input[1:]
            else:
                return (new_x,)
        
        return layer.register_forward_pre_hook(prompt_pre_hook)

    def _freeze_backbone(self):
        for param in self.convnet.parameters():
            param.requires_grad = False
        
        # 确保prompt参数是可训练的
        for prompt in self.visual_prompts:
            prompt.requires_grad = True
