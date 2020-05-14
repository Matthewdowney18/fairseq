import torch
import torch.nn.functional as F

import math

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('cross_entropy_head_diversification')
class KLHeadDiversificationCriterion(FairseqCriterion):

        def __init__(self, task, sentence_avg, kl_reg):
            super().__init__(task)
            self.sentence_avg = sentence_avg
            self.epsilon = kl_reg

        @staticmethod
        def add_args(parser):
            """Add criterion-specific arguments to the parser."""
            # fmt: off
            parser.add_argument('--kl-reg', default=0., type=float, metavar='D',
                                help='epsilon for kl regularization, 0 means no label smoothing')
            # fmt: on

        def forward(self, model, sample, reduce=True):
            """Compute the loss for the given sample.

            Returns a tuple with three elements:
            1) the loss
            2) the sample size, which is used as the denominator for the gradient
            3) logging outputs to display while training
            """
            net_output = model(**sample['net_input'])
            loss, kl_reg = self.compute_loss(model, net_output, sample, reduce=reduce)
            # sentence_avg is false
            sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
            logging_output = {
                'loss': loss.data,
                'ntokens': sample['ntokens'],
                'nsentences': sample['target'].size(0),
                'sample_size': sample_size,
                'kl_reg': kl_reg,
            }
            return loss, sample_size, logging_output

        def compute_loss(self, model, net_output, sample, reduce=True):
            scores, weights = net_output
            #needed to make the net output a tuple of scores and a dict which isnt used
            lprobs = model.get_normalized_probs((scores, None), log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = model.get_targets(sample, scores).view(-1)
            loss = F.nll_loss(
                lprobs,
                target,
                ignore_index=self.padding_idx,
                reduction='sum' if reduce else 'none',
            )

            # reward
            kl_reg = self.calculate_kl_div_reward(weights)
            loss -= kl_reg * self.epsilon
            return loss, kl_reg

        @staticmethod
        def reduce_metrics(logging_outputs) -> None:
            """Aggregate logging outputs from data parallel training."""
            loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
            kl_sum = sum(log.get('kl_reg', 0) for log in logging_outputs)
            ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
            sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
            # sample size is denominator for gradient
            metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
            # im not sure how correct this is
            metrics.log_scalar("kl_reg", kl_sum / sample_size, sample_size, round=10)
            if sample_size != ntokens:
                metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
                metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
            else:
                metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

        @staticmethod
        def logging_outputs_can_be_summed() -> bool:
            """
            Whether the logging outputs returned by `forward` can be summed
            across workers prior to calling `reduce_metrics`. Setting this
            to True will improves distributed training speed.
            """
            return True

        def calculate_kl_div_reward(self, weights):
            """weights = list(tesnor([num_heads, batch_size, dim_q, dim_k]))"""
            kl_scores = list()
            for weight in weights:
                # need to say what gpu to send tensor
                # concat is very bad
                # create python lists and concatinate at the end
                p_mats = list()
                q_mats = list()
                num_heads = weight.size(0)
                for i in range(0, num_heads):
                    for j in range(i + 1, num_heads):
                        p_mats.append(weight[i, :, :, :].unsqueeze(0))
                        q_mats.append(weight[j, :, :, :].unsqueeze(0))
                p_mat = torch.cat(p_mats, 0)
                q_mat = torch.cat(q_mats, 0)
                score = self.KL_div(p_mat, q_mat)
                kl_scores.append(score.unsqueeze(0))
            # already sum or tgt_seq
            # want to sum over src_seq, batch_sixe
            # want to take mean over num_combos and num_modules
            # want to divide by num_combos*num_modules
            return torch.sum(torch.cat(kl_scores, 0)) / len(kl_scores) / len(p_mats)

        @staticmethod
        # function to calculate the kl div of 2 tensors
        def KL_div(p, q):
            """
            P, Q are tensors of the same size, the last dims
            are probability distributions (add up to 1).
            Returns a tensor of the same shape with the las dim reduced.
            """
            assert (p.size() == q.size()), "p, q must be the same size"
            sum_p = torch.sum(p, -1)
            sum_q = torch.sum(q, -1)
            # need this fancy stuff to ignore small precision differences
            # anything greater than 1e-6 does not work with softmax
            assert torch.all(torch.lt(torch.abs(torch.add(sum_p, -torch.ones_like(sum_p))), 1e-6)), "p has at least 1 non prob distribution"
            assert torch.all(torch.lt(torch.abs(torch.add(sum_q, -torch.ones_like(sum_q))), 1e-6)), "q has at least 1 non prob distribution"

            # make sure no values in q are equal to 0
            # This seems to work nicely but im not sure how efficent
            q[q <= 1e-7] = 1e-7
            # I have not figured out how to divide by the sum yet
            # q /= q.sum(-1)

            #do not average because loss is not normalized
            return F.kl_div(torch.log(q), p, reduction="sum")
