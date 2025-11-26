import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import AdapterClipNet
from utils.inc_net import PromptClipNet
from utils.inc_net import LoRAClipNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, get_attribute, ClipLoss
from utils.data_manager import LaionData
import math
import os


num_workers = 8
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        peft_method=get_attribute(args, "peft_method", "adapter")
        # 使用 AdapterClipNet
        if peft_method=="adapter":
            self._network = AdapterClipNet(args, False)
        elif peft_method=="prompt":
            self._network = PromptClipNet(args,False)
        else:
            self._network=LoRAClipNet(args,False)
        
        self.batch_size = get_attribute(args, "batch_size", 48)
        self.init_lr = get_attribute(args, "init_lr", 0.01)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 1e-8)

        self.tuned_epoch = get_attribute(args, "tuned_epoch", 5)
        self._known_classes = 0

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        self._network.to(self._device)

        #训练数据：此处可见数据集是
        train_dataset = data_manager.get_dataset(np.arange(0, self._total_classes),source="train", mode="train", appendent=self._get_memory())
        self.data_manager = data_manager
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        #测试数据
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._train_clip_peft(self.train_loader, self.test_loader)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    
    def _train_clip_peft(self, train_loader, test_loader):
        
        self._network.to(self._device)
        
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam': 
            optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)

        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt[0]
        prog_bar = tqdm(range(self.tuned_epoch))
        cliploss = ClipLoss()

        total_labels = class_to_label[:self._total_classes]  # 所有已知类别
        
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                labels = [class_to_label[y] for y in targets]
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                
                # 准备所有类别的文本
                texts = [templates.format(inst) for inst in total_labels]
                texts = self._network.tokenizer(texts).to(self._device)
                
                # AdapterClipNet 前向传播
                image_features, text_features, logit_scale = self._network(inputs, texts)
                
                # 归一化特征
                img_feas = F.normalize(image_features, dim=-1)
                text_feas = F.normalize(text_features, dim=-1)
                
                # 计算logits
                logits = logit_scale * img_feas @ text_feas.T
                
                # CLIP损失 - 使用目标类别的文本特征
                target_texts = [templates.format(inst) for inst in labels]
                clip_text_feas = self._network.encode_text(self._network.tokenizer(target_texts).to(self._device))
                clip_text_feas = F.normalize(clip_text_feas, dim=-1)
                clip_loss = cliploss(img_feas, clip_text_feas, logit_scale)
                
                # 分类损失
                loss = F.cross_entropy(logits, targets)
                
                # 总损失
                total_loss = loss + clip_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                losses += total_loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                self._cur_task, epoch + 1, self.args['tuned_epoch'], losses / len(train_loader), train_acc, test_acc
            )
            prog_bar.set_description(info)

    def _compute_accuracy(self, model, loader):
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes]
        
        # 准备所有类别的文本特征
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).to(self._device)
                class_embeddings = self._network.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)
        
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                image_features = self._network.encode_image(inputs)
                image_features = F.normalize(image_features, dim=-1)
                text_features_norm = F.normalize(text_features, dim=-1)
                outputs = image_features @ text_features_norm.T

            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
            
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes]
        
        # 准备文本特征
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).to(self._device)
                class_embeddings = self._network.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)

        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                image_features = self._network.encode_image(inputs)
                image_features = F.normalize(image_features, dim=-1)
                text_features_norm = F.normalize(text_features, dim=-1)
                outputs = image_features @ text_features_norm.T

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)