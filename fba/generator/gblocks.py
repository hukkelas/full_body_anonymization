import torch


class MaskOutput(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x_real, x_fake, mask):
        ctx.save_for_backward(mask)
        out = x_real * mask + (1-mask) * x_fake
        return out

    @staticmethod
    def backward(ctx, grad_output):
        fake_grad = grad_output
        mask, = ctx.saved_tensors
        fake_grad = fake_grad * (1 - mask)
        known_percentage = mask.view(mask.shape[0], -1).mean(dim=1)
        fake_grad = fake_grad / (1-known_percentage).view(-1, 1, 1, 1)
        return None, fake_grad, None


def mask_output(scale_grad, x_real, x_fake, mask):
    if scale_grad:
        return MaskOutput.apply(x_real, x_fake, mask)
    return x_real * mask + (1-mask) * x_fake
