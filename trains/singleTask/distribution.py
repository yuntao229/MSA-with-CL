import torch

# Pull close postive batch and negative batch seperately.
def gussian_kld(m1, m2, v1, v2):
    kld = -0.5 * torch.sum(1 - torch.div(v1 ** 2, v2 ** 2)
                           - torch.div((m1 - m2) ** 2, v2 ** 2 - 2 * torch.log(torch.div(v2, v1))))
    return torch.mean(kld*0.02)
