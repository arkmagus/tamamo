import torch
from torch.nn import Module

class ElementwiseNLLLoss(Module) :
    def __init__(self, size_average=True, weight=None) :
        super(ElementwiseNLLLoss, self).__init__()
        self.size_average = size_average 
        self.weight = weight
    
    def forward(self, log_input, target) :
        # target -1 means ignored #
        log_input_target = log_input.gather(1, target.unsqueeze(1)).squeeze()

        loss = log_input_target

        if self.weight is not None :
            loss = loss * self.weight.index_select(0, target)

        loss = -loss # negative LL # 
        return loss
    pass

def elementwise_nllloss(input, target, size_average=True, weight=None) :
    return ElementwiseNLLLoss(size_average, weight)(input, target)

class ElementwiseCrossEntropy(Module) :
    def __init__(self, size_average=True, weight=None) :
        super(ElementwiseCrossEntropy, self).__init__()
        self.size_average = size_average
        self.weight = weight

    def forward(self, input, target) :
        return elementwise_nllloss(torch.nn.functional.log_softmax(input, dim=-1), target, 
                self.size_average, self.weight)
    pass

def elementwise_crossentropy(input, target, size_average=True, label_smoothing=0, weight=None) :
    return ElementwiseCrossEntropy(size_average, label_smoothing, weight)(input, target)

class ElementwiseBCE(Module) :
    def __init__(self) :
        super().__init__()
        pass

    def forward(self, input, target) :
        return -(target*torch.log(input) + (1-target)*torch.log(1-input))
    pass

def elementwise_bce(input, target) :
    return ElementwiseBCE()(input, target)

class ElementwiseBCEWithLogits(Module) :
    def __init__(self) :
        super().__init__()
        pass

    def forward(self, input, target) :
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        return loss

def elementwise_bce_with_logits(input, target) :
    return ElementwiseBCEWithLogits()(input, target)
