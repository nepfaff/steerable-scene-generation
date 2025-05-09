"""
Modified from the following:
https://github.com/MIT-SPARK/MiDiffusion/blob/main/midiffusion/networks/diffusion_d3pm.py
https://github.com/microsoft/VQ-Diffusion/blob/main/image_synthesis/modeling/transformers/diffusion_transformer.py

Note that cleaning this up is TBC due to time pressure. I'm sorry for the current state
of this...
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

EPS_PROB = 1e-30  # minimum probability to make sure log prob is numerically stable
LOG_ZERO = -69  # substitute of log(0)


class BaseDiffusion(nn.Module):
    """
    Base class for diffusion model.
    Note that theoretically t = 1, ..., num_steps, the function argument t
    starts from 0 (i.e. off by 1) due to python indexing.
    """

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def q_pred(self, x_0, t):
        """
        Compute probability q(x_t | x_0)
        """
        raise NotImplementedError

    def q_sample(self, x_0, t):
        """
        Diffuse the data, i.e. sample from q(x_t | x_0)
        """
        raise NotImplementedError

    def q_posterior(self, x_0, x_t, t):
        """
        Compute posterior probability q(x_{t-1} | x_t, x_0)
        """
        raise NotImplementedError

    def p_pred(self, denoise_fn, x_t, t, condition, **kwargs):
        """
        Compute denoising probability p(x_{t-1} | x_t)
        """
        raise NotImplementedError

    def p_sample(self, denoise_fn, x_t, t, condition, **kwargs):
        """
        Denoise the data, i.e. sample from p(x_{t-1} | x_t)
        """
        raise NotImplementedError

    def p_sample_loop(self, denoise_fn, shape, condition, sample_freq=None, **kwargs):
        """
        Generate data by denoising recursively
        """
        raise NotImplementedError

    def p_losses(self, denoise_fn, x_0, condition, **kwargs):
        """
        Training loss calculation
        """
        raise NotImplementedError

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def _move_tensors(self, device):
        """
        Move pre-computed parameters to specified device
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(device))
        self.device = torch.device(device)

    @staticmethod
    def sum_last_dims(data_tensor, keep_dims=1):
        """
        Sum over the last dimensions. Default: sum input data over each batch.
        """
        return data_tensor.reshape(*data_tensor.shape[:keep_dims], -1).sum(-1)

    @staticmethod
    def mean_per_batch(data_tensor, feat_size=None, start_ind=0, mask=None):
        """
        Given B x N x C input data, return a 1-D average tensor of size B,
        with option to specify a B x N mask or a feature range to select data.
        """
        B, N, C = data_tensor.shape

        # Handle the case where feat_size is zero
        if feat_size == 0:
            return torch.zeros(B, device=data_tensor.device)

        # Determine feature size if not specified
        if feat_size is None:
            feat_size = C - start_ind

        # Select the relevant feature range from the data
        data_selected = data_tensor[:, :, start_ind : start_ind + feat_size]

        # Compute mean with or without mask
        if mask is not None:
            assert mask.shape == (B, N) and (mask.sum(dim=1) > 0).all()
            masked_sum = (data_selected * mask.unsqueeze(-1)).sum(dim=[1, 2])
            return masked_sum / (mask.sum(dim=1) * feat_size)
        else:
            return data_selected.mean(dim=[1, 2])


def log_1_min_a(a):
    """log(1 - exp(a))"""
    return torch.log(1 - a.exp() + EPS_PROB)


def log_add_exp(a, b):
    """log(exp(a) + exp(b))"""
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def alpha_schedule(
    num_timesteps, N=100, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.99999
):
    # note: 0.0 will tends to raise unexpected behaviour (e.g., log(0.0)), thus avoid 0.0
    assert att_1 > 0.0 and att_T > 0.0 and ctt_1 > 0.0 and ctt_T > 0.0
    assert att_1 + ctt_1 <= 1.0 and att_T + ctt_T <= 1.0

    att = np.arange(0, num_timesteps) / (num_timesteps - 1) * (att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]
    ctt = np.arange(0, num_timesteps) / (num_timesteps - 1) * (ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct
    bt = (1 - at - ct) / N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1 - att - ctt) / N

    def _f(x):
        return torch.tensor(x.astype("float64"))

    return _f(at), _f(bt), _f(ct), _f(att), _f(btt), _f(ctt)


class MaskAndReplaceDiffusion(BaseDiffusion):
    def __init__(
        self,
        num_classes,  # Includes [mask] token
        num_timesteps: int = 1000,
        noise_params=None,
        model_output_type="x0",
        mask_weight=1,
        auxiliary_loss_weight=0,
        adaptive_auxiliary_loss=False,
    ):
        super().__init__()

        assert model_output_type in ["x0", "x_prev"]
        assert auxiliary_loss_weight >= 0
        assert mask_weight >= 0
        self.num_classes = (
            num_classes  # TODO: currently, this is includes 'empty' and 'mask'
        )
        self.num_timesteps = num_timesteps
        self.timesteps = torch.tensor(list(reversed(range(0, num_timesteps)))).long()
        self.model_output_type = model_output_type
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.mask_weight = mask_weight  # 'mask' token weight

        if noise_params is None:
            # Don't include [mask] token.
            noise_params = alpha_schedule(num_timesteps, num_classes - 1)

        # diffusion noise params
        at, bt, ct, att, btt, ctt = noise_params
        assert at.shape[0] == bt.shape[0] == ct.shape[0]
        assert att.shape[0] == btt.shape[0] == ctt.shape[0] == at.shape[0] + 1
        self.num_timesteps = at.shape[0]

        log_at, log_bt, log_ct = torch.log(at), torch.log(bt), torch.log(ct)
        log_cumprod_at, log_cumprod_bt, log_cumprod_ct = (
            torch.log(att),
            torch.log(btt),
            torch.log(ctt),
        )

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.0e-5
        assert (
            log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item()
            < 1.0e-5
        )

        self.diffusion_acc_list = [0] * self.num_timesteps
        self.diffusion_keep_list = [0] * self.num_timesteps
        # Convert to float32 and register buffers.
        self.register_buffer("log_at", log_at.float())
        self.register_buffer("log_bt", log_bt.float())
        self.register_buffer("log_ct", log_ct.float())
        self.register_buffer("log_cumprod_at", log_cumprod_at.float())
        self.register_buffer("log_cumprod_bt", log_cumprod_bt.float())
        self.register_buffer("log_cumprod_ct", log_cumprod_ct.float())
        self.register_buffer("log_1_min_ct", log_1_min_ct.float())
        self.register_buffer("log_1_min_cumprod_ct", log_1_min_cumprod_ct.float())

        self.register_buffer("Lt_history", torch.zeros(self.num_timesteps))
        self.register_buffer("Lt_count", torch.zeros(self.num_timesteps))

    def multinomial_kl(self, log_prob1, log_prob2):  # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    @staticmethod
    def log_onehot_to_index(x: torch.FloatTensor) -> torch.LongTensor:
        """
        Convert (B, C, N) log one-hot tensor to (B, N) index tensor. This basically
        applies `argmax(log_x, dim=1)`.

        Args:
            x (torch.FloatTensor): Input tensor of shape (B, C, N) where C are the
                number of discrete categories and is the length of the one-hot vector.

        Return:
            torch.LongTensor: The index tensor whose non-negative int values indicate
                the category. Has shape (B, N).
        """
        return x.argmax(dim=1)

    @staticmethod
    def onehot_to_index(log_x: torch.FloatTensor) -> torch.LongTensor:
        """
        Convert (B, C, N) one-hot tensor to (B, N) index tensor. This basically
        applies `argmax(log_x, dim=1)`.

        Args:
            log_x (torch.FloatTensor): Input tensor of shape (B, C, N) where C are the
                number of discrete categories and is the length of the one-hot vector.

        Return:
            torch.LongTensor: The index tensor whose non-negative int values indicate
                the category. Has shape (B, N).
        """
        return log_x.argmax(dim=1)

    @staticmethod
    def index_to_log_onehot(x: torch.LongTensor, num_classes: int) -> torch.FloatTensor:
        """
        Convert (B, N) index tensor to (B, C, N) log one-hot tensor.

        Args:
            x (torch.LongTensor): Input tensor of shape (B, N) with non-negative int
                values that indicate the category.
            num_classes (int): The number of discrete classes (length of the resulting)
                one-hot vector.

        Return:
            torch.FloatTensor: The input in log one-hot form of shape (B, C, N) where
                C = num_classes.
        """
        assert x.max().item() < num_classes, f"Error: {x.max().item()} >= {num_classes}"
        x_onehot = F.one_hot(x, num_classes)
        permute_order = (0, -1) + tuple(range(1, x.ndim))
        x_onehot = x_onehot.permute(permute_order)
        log_x = torch.log(x_onehot.float().clamp(min=EPS_PROB))
        return log_x

    @staticmethod
    def onehot_to_log_onehot(x_onehot: torch.LongTensor) -> torch.FloatTensor:
        """
        Convert (B, N, C-1) one-hot tensor to (B, C, N) log one-hot tensor.

        Args:
            x_onehot (torch.LongTensor): Input one-hot tensor of shape (B, N, C-1).

        Return:
            torch.FloatTensor: The input in log one-hot form of shape (B, C, N).
        """
        # Add a zero vector to extend the one-hot encoding to (B, N, C).
        zeros = torch.zeros(
            *x_onehot.shape[:-1], 1, dtype=x_onehot.dtype, device=x_onehot.device
        )
        extended_onehot = torch.cat((x_onehot, zeros), dim=-1)

        # Permute to (B, C, N).
        permute_order = (0, 2, 1)
        extended_onehot = extended_onehot.permute(permute_order)

        # Convert to log one-hot form,
        log_x = torch.log(extended_onehot.float().clamp(min=EPS_PROB))

        return log_x

    def q_pred_one_timestep(self, log_x_t_1, t):
        """
        log(Q_t * exp(log_x_t_1)), diffusion step: q(x_t | x_{t-1})
        """
        # log_x_t_1 (B, C, N)
        log_at = self._extract(self.log_at, t, log_x_t_1.shape)  # at
        log_bt = self._extract(self.log_bt, t, log_x_t_1.shape)  # bt
        log_ct = self._extract(self.log_ct, t, log_x_t_1.shape)  # ct
        log_1_min_ct = self._extract(self.log_1_min_ct, t, log_x_t_1.shape)  # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(
                    log_x_t_1[:, :-1, :] + log_at, log_bt
                ),  # dropped a small term
                log_add_exp(log_x_t_1[:, -1:, :] + log_1_min_ct, log_ct),
            ],
            dim=1,
        )

        return log_probs

    def q_pred(self, log_x_start, t):
        """
        log(bar{Q}_t * exp(log_x_start)), diffuse the data to time t: q(x_t | x_0)
        """
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        log_cumprod_at = self._extract(self.log_cumprod_at, t, log_x_start.shape)  # at~
        log_cumprod_bt = self._extract(self.log_cumprod_bt, t, log_x_start.shape)  # bt~
        log_cumprod_ct = self._extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        log_1_min_cumprod_ct = self._extract(
            self.log_1_min_cumprod_ct, t, log_x_start.shape
        )  # 1-ct~

        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:, :-1, :] + log_cumprod_at, log_cumprod_bt),
                log_add_exp(
                    log_x_start[:, -1:, :] + log_1_min_cumprod_ct, log_cumprod_ct
                ),  # simplified
            ],
            dim=1,
        )

        return log_probs

    def q_posterior(
        self, log_x0_recon: torch.FloatTensor, log_x_t: torch.FloatTensor, t
    ) -> torch.FloatTensor:
        """
        Returns the log of prosterior probability q(x_{t-1}|x_t,x_0').

        Args:
            log_x0_recon (torch.FloatTensor): The log probabilities for the predicted
                classes of the initial categories. Has shape (B, C, N) where C is the
                number of classes, including the [mask] token.
            log_x_t (torch.FloatTensor): The log probabilities of the distribution at
                denoising timestep t of shape (B, C, N).
            t (torch.LongTensor): The timestep that determines which transition matrix
                is used.

        Return:
            torch.FloatTensor: The log probabilities for the predicted classes at
                timestep `t-1` of shape (B, C, N).
        """
        B, C, N = log_x0_recon.shape
        log_one_vector = torch.zeros(B, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.full((B, 1, N), LOG_ZERO).type_as(log_x_t)

        # notice that log_x_t is onehot
        onehot_x_t = self.log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes - 1).unsqueeze(1)

        log_qt = self.q_pred(log_x_t, t)  # q(xt|x0)
        # log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1)
        log_qt = log_qt[:, :-1, :]
        log_cumprod_ct = self._extract(
            self.log_cumprod_ct, t, log_x0_recon.shape
        )  # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes - 1, -1)
        # ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)  # q(xt|xt_1)
        log_qt_one_timestep = torch.cat(
            (log_qt_one_timestep[:, :-1, :], log_zero_vector), dim=1
        )
        log_ct = self._extract(self.log_ct, t, log_x0_recon.shape)  # ct
        ct_vector = log_ct.expand(-1, self.num_classes - 1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

        # log_x_start = torch.cat((log_x_start, log_zero_vector), dim=1)
        # q = log_x_start - log_qt
        q = log_x0_recon[:, :-1, :] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = (
            self.q_pred(q, t - 1) + log_qt_one_timestep + q_log_sum_exp
        )

        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, LOG_ZERO, 0)

    @staticmethod
    def log_pred_from_denoise_out(denoise_out: torch.FloatTensor) -> torch.FloatTensor:
        """
        Convert output of denoising network to log probability over classes and [mask].
        This is basically applying log softmax and concatenates the missing dimension
        for the [mask] token.

        Args:
            denoise_out (torch.FloatTensor): The output of the denoising network that
                does not necessarily correspond to probabilities. This has shape (B,
                C-1, N) where C-1 is the number of discrete classes without the [mask]
                token.

        Return:
            torch.FloatTensor: A log probability over the classes and [mask]. This has
                shape (B, C, N) where C includes the [mask] token.
        """
        B, _, N = denoise_out.shape

        log_pred = F.log_softmax(denoise_out.double(), dim=1).float()
        log_pred = torch.clamp(log_pred, LOG_ZERO, 0)
        log_zero_vector = torch.full((B, 1, N), LOG_ZERO).type_as(log_pred)
        return torch.cat((log_pred, log_zero_vector), dim=1)

    def predict_denoise(
        self, denoise_fn, log_x_t, t, condition=None, condition_cross=None
    ):
        """
        compute denoise_fn(x_t, t, context, context_cross) and convert output to log prob
        """
        x_t = self.log_onehot_to_index(log_x_t)  # (B, N)

        out = denoise_fn(x_t, t, context=condition, context_cross=condition_cross)
        log_pred = self.log_pred_from_denoise_out(out)
        assert log_pred.shape == log_x_t.shape

        return log_pred

    def p_pred(self, denoise_fn, log_x_t, t, condition=None, condition_cross=None):
        """
        log denoising probability, denoising step: p(x_{t-1} | x_t)
        """
        if self.model_output_type == "x0":
            # if x0, first p(x0|xt), than sum(q(xt-1|xt,x0)*p(x0|xt))
            log_x_recon = self.predict_denoise(
                denoise_fn, log_x_t, t, condition, condition_cross
            )
            log_model_pred = self.q_posterior(
                log_x0_recon=log_x_recon, log_x_t=log_x_t, t=t
            )
            return log_model_pred, log_x_recon
        elif self.model_output_type == "x_prev":
            log_model_pred = self.predict_denoise(
                denoise_fn, log_x_t, t, condition, condition_cross
            )
            return log_model_pred, None
        else:
            raise NotImplemented

    def q_sample_one_step(self, log_x_t_1, t, no_mask=False):
        """
        sample from q(x_t | x_{t-1})
        """
        log_EV_qxt = self.q_pred_one_timestep(log_x_t_1, t)
        log_sample = self.log_sample_categorical(log_EV_qxt, no_mask)
        return log_sample

    def q_sample(
        self, log_x_0: torch.FloatTensor, t: torch.LongTensor, no_mask=False
    ) -> torch.FloatTensor:
        """
        Sample from the forward process q(x_t | x_0) and return the log probability.

        Args:
            log_x_0 (torch.FloatTensor): The log form of the clean categories of
                shape (B, C, N) where C is the number of classes. This has one-hot
                vector form.
            t (torch.LongTensor): The timestep that determines the amount of noise to
                add of shape (B,).

        Return:
            torch.FloatTensor: The log form of the noisy categories x_t of shape (B, C,
                N). This has one-hot vector form.
        """
        log_EV_qxt_x0 = self.q_pred(log_x_0, t)
        log_x_t = self.log_sample_categorical(log_EV_qxt_x0, no_mask)
        return log_x_t

    @torch.no_grad()
    def p_sample(self, denoise_fn, log_x_t, t, condition, condition_cross=None):
        """
        sample x_{t-1} from p(x_{t-1} | x_t)
        """
        model_log_prob, _ = self.p_pred(
            denoise_fn, log_x_t, t, condition, condition_cross
        )
        log_sample = self.log_sample_categorical(model_log_prob)
        return log_sample

    def log_sample_categorical(
        self, log_props: torch.FloatTensor, no_mask: bool = False
    ) -> torch.FloatTensor:
        """
        Sample from a log probability under gumbel noise, return results as log of a
        one-hot embedding.

        Args:
            log_props (torch.FloatTensor): The log probabilities to sample from of shape
                (B, C, N).
            no_mask (bool): Whether to sample without the last [mask] token class.

        Return:
            torch.FloatTensor: The sample in log one-hot form of shape (B, C, N).
        """
        # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(log_props)
        gumbel_noise = -torch.log(-torch.log(uniform + EPS_PROB) + EPS_PROB)
        if no_mask:
            sample = (gumbel_noise + log_props)[:, :-1, :].argmax(dim=1)
        else:
            sample = (gumbel_noise + log_props).argmax(dim=1)
        log_sample = self.index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def sample_time(self, b, device, method="uniform"):
        if method == "importance":
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method="uniform")

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)
            pt = pt_all.gather(dim=0, index=t)
        elif method == "uniform":
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
        else:
            raise ValueError
        return t, pt

    def compute_kl_loss(
        self,
        log_x_0: torch.FloatTensor,
        log_x_t: torch.FloatTensor,
        t: torch.LongTensor,
        log_pred_prob: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Computes the KL-divergence training loss.

        Args:
            log_x_0 (torch.FloatTensor): The actual initial categories of shape (B, C,
                N).
            log_x_t (torch.FloatTensor): The log probabilities of the distribution at
                denoising timestep t of shape (B, C, N).
            t (torch.LongTensor): The timestep that determines which transition matrix
                is used.
            log_pred_prob (torch.FloatTensor): The log probabilities for the predicted
                classes at timestep `t-1` of shape (B, C, N).

        Return:
            torch.FloatTensor: The KL-divergence loss for each element in the batch of
                shape (B,).
        """
        log_q_prob = self.q_posterior(log_x_0, log_x_t, t)
        kl = self.multinomial_kl(log_q_prob, log_pred_prob)
        decoder_nll = -log_categorical(log_x_0, log_pred_prob)

        t0_mask = (t == 0).unsqueeze(1).repeat(1, log_x_0.shape[-1])
        kl_loss = torch.where(t0_mask, decoder_nll, kl)
        return kl_loss

    def compute_aux_loss(
        self,
        log_x_0: torch.FloatTensor,
        log_x0_recon: torch.FloatTensor,
        t: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Compute the auxilary loss which regulates the predicted x0.

        Args:
            log_x_0 (torch.FloatTensor): The actual initial categories of shape (B, C,
                N).
            log_x0_recon (torch.FloatTensor): The log probabilities for the predicted
                classes of the initial categories. Has shape (B, C, N) where C is the
                number of classes, including the [mask] token.
            t (torch.LongTensor): The timestep that resulted in log_x0_recon of shape
                (B,).

        Return:
            torch.FloatTensor: The auxiliary loss for each element in the batch of
                shape (B,).
        """
        aux_loss = self.multinomial_kl(log_x_0[:, :-1, :], log_x0_recon[:, :-1, :])

        t0_mask = (t == 0).unsqueeze(1).repeat(1, log_x_0.shape[-1])
        aux_loss = torch.where(t0_mask, torch.zeros_like(aux_loss), aux_loss)
        return aux_loss

    def p_losses(self, denoise_fn, x_start, t=None, pt=None, condition=None):
        assert self.model_output_type == "x0"
        if t is None or pt is None:
            t, pt = self.sample_time(x_start.size(0), x_start.device, "uniform")

        log_xstart = self.index_to_log_onehot(x_start, self.num_classes)
        log_xt = self.q_sample(log_x_0=log_xstart, t=t)

        log_model_prob, log_x0_recon = self.p_pred(denoise_fn, log_xt, t, condition)

        x0_recon = self.log_onehot_to_index(log_x0_recon)
        x0_real = x_start
        xt_1_recon = self.log_onehot_to_index(log_model_prob)
        xt_recon = self.log_onehot_to_index(log_xt)
        for index in range(t.size(0)):
            this_t = t[index].item()
            same_rate = (x0_recon[index] == x0_real[index]).sum().cpu() / x0_real.size(
                1
            )
            self.diffusion_acc_list[this_t] = (
                same_rate.item() * 0.1 + self.diffusion_acc_list[this_t] * 0.9
            )
            same_rate = (
                xt_1_recon[index] == xt_recon[index]
            ).sum().cpu() / xt_recon.size(1)
            self.diffusion_keep_list[this_t] = (
                same_rate.item() * 0.1 + self.diffusion_keep_list[this_t] * 0.9
            )

        # Compute train loss
        loss_tensor = self.compute_kl_loss(log_xstart, log_xt, t, log_model_prob)
        if self.mask_weight != 1:  # adjust [mask] token weight
            mask_region = self.log_onehot_to_index(log_xt) == self.num_classes - 1
            loss_tensor = torch.where(
                mask_region, self.mask_weight * loss_tensor, loss_tensor
            )
        kl_loss = self.sum_last_dims(loss_tensor, keep_dims=1)

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # Upweigh loss term of the kl
        loss1 = kl_loss / pt
        vb_loss = loss1.mean()
        losses_dict = {"kl_loss": loss1.mean()}

        if self.auxiliary_loss_weight > 0:
            loss_tensor = self.compute_aux_loss(log_xstart, log_x0_recon, t)
            if self.mask_weight != 1:  # adjust [mask] token weight
                loss_tensor = torch.where(
                    mask_region, self.mask_weight * loss_tensor, loss_tensor
                )
            aux_loss = self.sum_last_dims(loss_tensor, keep_dims=1)
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1 - t / self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * aux_loss / pt
            losses_dict["aux_loss"] = loss2.mean()
            vb_loss += loss2.mean()

        return vb_loss, losses_dict

    def p_sample_loop(
        self, denoise_fn, log_x_end, condition, condition_cross=None, sample_freq=None
    ):
        B, C, N = log_x_end.shape
        assert C == self.num_classes
        if sample_freq:
            pred_traj = [self.log_onehot_to_index(log_x_end)]

        log_x_t = log_x_end
        total_steps = self.num_timesteps
        for t in reversed(range(0, total_steps)):
            t_ = torch.full((B,), t, dtype=torch.int64, device=self.device)
            log_x_t = self.p_sample(
                denoise_fn=denoise_fn,
                log_x_t=log_x_t,
                t=t_,
                condition=condition,
                condition_cross=condition_cross,
            )  # log_x_t is log_onehot
            if sample_freq and (t % sample_freq == 0 or t == total_steps - 1):
                pred_traj.append(self.log_onehot_to_index(log_x_t))

        if sample_freq:
            return pred_traj
        else:
            return self.log_onehot_to_index(log_x_t)
