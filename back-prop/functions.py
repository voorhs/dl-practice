from torch.autograd import Function
import torch
import torch.nn.functional as F 


class ReLUFunction(Function):
    @staticmethod
    def forward(ctx, input):
        mask = input > 0
        output = torch.where(mask, input, 0)
        ctx.save_for_backward(mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[~mask] = 0
        return grad_input


class LinearFunction(Function):
    @staticmethod
    def forward(ctx, inp, weight, bias):
        """
        inp: (B, in_features)
        weight: (out_features, in_features)
        bias: (out_features,)
        """
        ctx.save_for_backward(inp, weight)
        output = inp.matmul(weight.T).add(bias)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        inp, weight = ctx.saved_tensors
        grad_input = grad_output.matmul(weight)
        grad_weight = grad_output.T.matmul(inp)
        grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias


class LogSoftmaxFunction(Function):
    @staticmethod
    def forward(ctx, inp):
        """
        inp: (B, n_features)
        """
        output = inp - torch.logsumexp(inp, dim=1, keepdim=True)
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_input = grad_output - grad_output.sum(dim=1, keepdim=True) * torch.exp(output)
        return grad_input


class DropoutFunction(Function):
    @staticmethod
    def forward(ctx, inp, p, training):
        """
        inp: (B, features)
        p: float from (0, 1)
        training: True = train mode, False = inference mode
        """
        ctx.training = training
        
        if not training:
            return inp

        mask = torch.rand_like(inp) < p
        ctx.save_for_backward(mask)
        ctx.p = p

        output = inp.clone()
        output[mask] = 0
        output[~mask] /= 1 - p

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()

        if not ctx.training:
            return grad_input

        mask, = ctx.saved_tensors
        
        grad_input[mask] = 0
        grad_input[~mask] /= 1 - ctx.p

        return grad_input, None, None


class CrossEntropyFunction(Function):
    @staticmethod
    def forward(ctx, activations: torch.Tensor, target, reduction='none'):
        ctx.save_for_backward(activations, target)
        ctx.reduction = reduction

        dummy = torch.arange(len(target), device=activations.device)
        
        nll = activations[dummy, target].neg()
        if reduction == 'sum':
            nll = nll.sum()
        elif reduction == 'mean':
            nll = nll.mean()
        
        return nll

    @staticmethod
    def backward(ctx, grad_output):
        activations, target = ctx.saved_tensors
        
        grad_output_clone = grad_output.clone()
        grad_input = torch.zeros_like(activations)
        grad_input[torch.arange(len(target)), target] = grad_output_clone.neg()

        if ctx.reduction == 'mean':
            grad_input /= len(target)

        return grad_input, None, None


class SiLUFunction(Function):
    @staticmethod
    def forward(ctx, input):
        sigmoid = 1 / (1 + torch.exp(-input))
        output = input * sigmoid
        ctx.save_for_backward(input, sigmoid)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, sigmoid = ctx.saved_tensors
        grad_f_x = sigmoid * (1 + input * (1 - sigmoid))
        grad_input = grad_f_x * grad_output
        return grad_input


class HardswishFunction(Function):
    @staticmethod
    def forward(ctx, input):
        mask_left = (-3 > input)
        output = torch.where(mask_left, 0, input)
        mask_inner = (~mask_left) & (input < 3)
        tmp = output[mask_inner]
        output[mask_inner] = tmp * (tmp + 3) / 6
        ctx.save_for_backward(mask_left, mask_inner, input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask_left, mask_inner, input = ctx.saved_tensors
        relu6 = input + 3
        relu6[mask_left] = 0
        relu6[~(mask_inner|mask_left)] = 1
        tmp = input * mask_inner
        grad_f_x = (relu6 + tmp) / 6
        grad_input = grad_f_x * grad_output
        return grad_input
