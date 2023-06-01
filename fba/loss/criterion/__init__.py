from .build import build_criterion
from .adversarial import wgan_d_loss, wgan_g_loss, nsgan_d_loss, nsgan_g_loss 
from .regularizer import improved_gradient_penalty, epsilon_penalty
from .unet import oasis_d_loss, oasis_g_loss
from .cse_regress import fpn_cse_d_loss, fpn_cse_g_loss