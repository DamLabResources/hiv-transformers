# Any Python script in the scripts folder will be able to import from this module.
import torch
from transformers import Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, mean_absolute_error, max_error, r2_score
from scipy.special import softmax


class CustomTrainer(Trainer):
    def __init__(self, *args, train_type = 'categorical', class_weights = None, **kwargs):
        class_weights = torch.tensor(class_weights) if class_weights is not None else torch.tensor(1)
        
        if train_type == 'multi_class':
            self.loss_fct = torch.nn.BCEWithLogitsLoss()#pos_weight = class_weights.to('cuda'))
        elif train_type == 'categorical':
            self.loss_fct = torch.nn.CrossEntropyLoss(weight = class_weights.to('cuda'))
        elif train_type == 'regression':
            self.loss_fct = torch.nn.MSELoss()
        else:
            raise ValueError(f'Did not understand train_type {train_type}')
        
        self.class_weights = class_weights
        self.train_type = train_type
        super(). __init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits').view(-1, self.model.config.num_labels)
        
        # adjust targets as needed
        if self.train_type == 'multi_class':
            targets = torch.flatten(labels).float().view(-1, self.model.config.num_labels)
        elif self.train_type == 'regression':
            targets = torch.flatten(labels).float()
            logits = torch.flatten(logits).float()
            
        else:
            targets = labels.view(-1)
        
        # compute custom loss
        
        loss = self.loss_fct(logits, targets)
        return (loss, outputs) if return_outputs else loss

    
    
    
    
def categorical_metrics_factory(id2label):
    
    num_labels = len(id2label)
    
    def compute_cat_metrics(pred):
    
        labels = pred.label_ids
        preds = pred.predictions.argmax(axis=1)    

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds)
        acc = accuracy_score(labels, preds)

        mets = {'acc': acc}
        for _id in range(num_labels):
            mets[id2label[_id] + '__' + 'f1'] = f1[_id]
            mets[id2label[_id] + '__' + 'precision'] = precision[_id]
            mets[id2label[_id] + '__' + 'recall'] = recall[_id]

        return mets
    
    return compute_cat_metrics


def multiclass_metrics_factory(id2label):
    
    num_labels = len(id2label)
    
    def compute_multiclass_metrics(pred):
    
        labels = pred.label_ids.reshape(-1, num_labels)
        preds = 1/(1 + np.exp(-pred.predictions))

        mets = {}
        for _id in range(num_labels):

            auc = roc_auc_score(labels[:, _id], preds[:, _id])
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels[:, _id], preds[:, _id]>0.5, average='binary')
            acc = accuracy_score(labels[:, _id], preds[:, _id]>0.5)

            mets[id2label[_id] + '__' + 'acc'] = acc
            mets[id2label[_id] + '__' + 'f1'] = f1
            mets[id2label[_id] + '__' + 'precision'] = precision
            mets[id2label[_id] + '__' + 'recall'] = recall
            mets[id2label[_id] + '__' + 'auc'] = auc
        return mets
    
    return compute_multiclass_metrics
    
    
def regression_metrics_factory(id2label):
    
    num_labels = len(id2label)
    
    def compute_reg_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions

        mae = mean_absolute_error(labels, preds)
        r2 = r2_score(labels, preds)
        merror = max_error(labels, preds)

        return {
            'mae': mae,
            'r2': r2,
            'max_error': merror
        }
    
    return compute_reg_metrics