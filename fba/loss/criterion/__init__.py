from .build import build_criterion
from .adversarial import wgan_d_loss, wgan_g_loss, nsgan_d_loss, nsgan_g_loss 
from .generator_criterions import discriminator_feature_matching_loss, masked_l1_loss, gaussian_kl_loss
from .regularizer import improved_gradient_penalty, epsilon_penalty
from .unet import oasis_d_loss, oasis_g_loss
from .cse_regress import fpn_cse_d_loss, fpn_cse_g_loss