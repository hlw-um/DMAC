import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.individual_mixer import  IndividualMixer
import torch as th
from torch.optim import RMSprop,Adam
import torch.nn.functional as F
import numpy as np
class QRLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.scheme = scheme
        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.N_QUANT = self.args.n_agents

        self.QUANTS = np.linspace(0.0, 1.0, self.N_QUANT + 1)[1:]
        self.QUANTS_TARGET = (np.linspace(0.0, 1.0, self.N_QUANT + 1)[:-1] + self.QUANTS)/2

        self.INV_QUANTS_TARGET = 1.0 - self.QUANTS_TARGET


        self.vdn_mixer = VDNMixer()

        self.mixer = IndividualMixer(args)

        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        #self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.optimiser = Adam(params=self.params, lr=args.lr, eps=args.optim_eps)
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        new_chosen_action_qvals, attend_mag_regs = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # # Mask out unavailable actions

        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        new_target_max_qvals, _ = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
        
        chosen_action_qvals = new_chosen_action_qvals
        target_max_qvals = new_target_max_qvals


        vdn_chosen_action_qvals = self.vdn_mixer(chosen_action_qvals, batch["state"][:, :-1])
        vdn_target_max_qvals = self.vdn_mixer(target_max_qvals, batch["state"][:, 1:])
        

        targets =  self.args.gamma * (1 - terminated) * target_max_qvals + rewards /self.args.n_agents

        vdn_target = self.args.gamma * (1 - terminated) * vdn_target_max_qvals + rewards 


        td_error = (vdn_chosen_action_qvals - vdn_target.detach())
        td_mask = mask.expand_as(td_error)

        masked_td_error = td_error * td_mask
        td_loss = (masked_td_error ** 2).sum() / td_mask.sum()


        # Td-error
        chosen_action_qvals = th.reshape(chosen_action_qvals, [-1, chosen_action_qvals.shape[-1]])
        targets = th.reshape(targets.detach(), [-1, targets.shape[-1]])


        chosen_action_qvals, chosen_action_qvals_index = th.sort(chosen_action_qvals)

        targets, targets_index = th.sort(targets)

        nb_agent = self.args.n_agents

        theta_loss_tile = th.unsqueeze(chosen_action_qvals, axis=2).repeat(1,1,nb_agent)   
        logit_valid_tile = th.unsqueeze(targets, axis=1).repeat(1,nb_agent,1)  

        
        tau = th.FloatTensor(self.QUANTS_TARGET).view(1, -1, 1)
        inv_tau  = th.FloatTensor(self.INV_QUANTS_TARGET).view(1, -1, 1)

        if self.args.use_cuda:
            tau = tau.cuda()
            inv_tau = inv_tau.cuda()

        tau = tau.detach()
        inv_tau = inv_tau.detach()

        error_loss = (logit_valid_tile - theta_loss_tile).detach()


        mask2 = th.reshape(mask, [-1,1,1])
        mask3 = mask2.expand_as(error_loss)
        Huber_loss = F.smooth_l1_loss(theta_loss_tile, logit_valid_tile, reduction='none')
        

        pair_wise_qr_loss = th.where(th.less(error_loss, 0.0), tau * Huber_loss, inv_tau * Huber_loss) * mask3


        # * self.args.n_agents is for reducing the term 1/N^{2}, so that we don't need to multiply 1/N^{2} for TD
        qr_loss =  (th.sum(th.mean(pair_wise_qr_loss,dim=2), dim = 1) * self.args.n_agents).sum()  / mask3.sum()

        loss = qr_loss + td_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num



    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
