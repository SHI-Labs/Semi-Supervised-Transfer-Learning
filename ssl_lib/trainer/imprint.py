
import os
import torch
import torch.nn as nn
import torch.nn.parallel


def imprint(model,loader,num_class,num_labels, device):
    print('Imprint the classifier of the model')
    model.eval()
    feat_size = 256
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # compute output
            out_list = model(inputs,return_fmap=True)
            output = out_list[-2]
            if batch_idx == 0:
                output_stack = output
                target_stack = targets
                feat_size=output.size(1)
            else:
                output_stack = torch.cat((output_stack, output), 0)
                target_stack = torch.cat((target_stack, targets), 0)
            if len(output_stack)>=num_labels and num_labels>0:
                output_stack=output_stack[:num_labels]
                target_stack=target_stack[:num_labels]
                break
    
    new_weight = torch.zeros(num_class, feat_size).to(device)
    for i in range(num_class):
        tmp = output_stack[target_stack == i].mean(0)
        new_weight[i] = tmp / tmp.norm(p=2)
    try:
        model.fc.weight.data = new_weight
    except:
        model.classifier.weight.data = new_weight
    model.train()
    return model