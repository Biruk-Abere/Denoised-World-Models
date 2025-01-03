import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


OUT_DIM = {2: 39, 4: 35, 6: 31}

class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim * 2)
        self.ln = nn.LayerNorm(self.feature_dim * 2)
        self.combine = nn.Linear(self.feature_dim + 2, self.feature_dim)

        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        #out = torch.tanh(h_norm)

        mu, logstd = torch.chunk(h_norm, 2, dim=-1)
        logstd = torch.tanh(logstd)
        self.outputs['mu'] = mu
        self.outputs['logstd'] = logstd
        self.outputs['std'] = logstd.exp()

        out = self.reparameterize(mu, logstd)
        self.outputs['tanh'] = out
        return out, mu, logstd

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)

class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters
    )

def club_loss(x_samples, x_mu, x_logvar, y_samples):        
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        
        positive = -(x_mu - y_samples)**2 / x_logvar.exp()
        negative = - (x_mu - y_samples[random_index])**2 / x_logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.


class TransitionModel(nn.Module):
    def __init__(self, state_size, hidden_size, action_size, history_size):
        super().__init__()

        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.history_size = history_size
        self.act_fn = nn.ELU()

        self.fc_state_action = nn.Linear(state_size + action_size, hidden_size)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.history_cell = nn.GRUCell(hidden_size, history_size)
        self.fc_state_mu = nn.Linear(history_size + hidden_size, state_size)
        self.fc_state_sigma = nn.Linear(history_size + hidden_size, state_size)

        self.min_sigma = 1e-4
        self.max_sigma = 1e0

    def init_states(self, batch_size, device):
        self.prev_state = torch.zeros(batch_size, self.state_size).to(device)
        self.prev_action = torch.zeros(batch_size, self.action_size).to(device)
        self.prev_history = torch.zeros(batch_size, self.history_size).to(device)
            
    def get_dist(self, mean, std):
        distribution = torch.distributions.Normal(mean, std)
        return distribution

    def stack_states(self, states, dim=0):        
        s = dict(
            mean = torch.stack([state['mean'] for state in states], dim=dim),
            std  = torch.stack([state['std'] for state in states], dim=dim),
            sample = torch.stack([state['sample'] for state in states], dim=dim),
            history = torch.stack([state['history'] for state in states], dim=dim),)
        if 'distribution' in states:
            dist = dict(distribution = [state['distribution'] for state in states])
            s.update(dist)
        return s
    
    def seq_to_batch(self, state, name):
        return dict(
            sample = torch.reshape(state[name], (state[name].shape[0]* state[name].shape[1], *state[name].shape[2:])))

    def transition_step(self, state, action, hist, not_done):
        state = state * not_done
        hist = hist * not_done

        state_action_enc = self.act_fn(self.fc_state_action(torch.cat([state, action], dim=-1)))
        state_action_enc = self.act_fn(self.fc_hidden(state_action_enc))
        state_action_enc = self.act_fn(self.fc_hidden(state_action_enc))
        state_action_enc = self.act_fn(self.fc_hidden(state_action_enc))

        current_hist = self.history_cell(state_action_enc, hist)
        next_state_mu = self.act_fn(self.fc_state_mu(torch.cat([state_action_enc, hist], dim=-1)))
        next_state_sigma = torch.tanh(self.fc_state_sigma(torch.cat([state_action_enc, hist], dim=-1)))
        next_state = next_state_mu + torch.randn_like(next_state_mu) * next_state_sigma.exp()

        state_enc = {"mean": next_state_mu, "logvar": next_state_sigma, "sample": next_state, "history": current_hist}
        return state_enc

    def observe_rollout(self, rollout_states, rollout_actions, init_history, nonterms):
        observed_rollout = []
        for i in range(rollout_states.shape[0]):
            rollout_states_ = rollout_states[i]
            rollout_actions_ = rollout_actions[i]
            init_history_ = nonterms[i] * init_history
            state_enc = self.observe_step(rollout_states_, rollout_actions_, init_history_)
            init_history = state_enc["history"]
            observed_rollout.append(state_enc)
        observed_rollout = self.stack_states(observed_rollout, dim=0)
        return observed_rollout
    
    def forward(self, state, action, hist, not_done):
        return self.transition_step(state, action, hist, not_done)

    def reparameterize(self, mean, std):
        eps = torch.randn_like(mean)
        return mean + eps * std
    
def club_loss(x_samples, x_mu, x_logvar, y_samples):        
        """
        This function computes the Contrastive Log Upper Bound (CLUB) loss.

        x_samples: (batch_size, sample_size, state_size)
        x_mu: (batch_size, state_size)
        x_logvar: (batch_size, state_size)
        y_samples: (batch_size, sample_size, state_size)
        """
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        
        positive = -(x_mu - y_samples)**2 / x_logvar.exp()
        negative = - (x_mu - y_samples[random_index])**2 / x_logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.0