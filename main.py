import hmm as gmove
import numpy as np

X_example = np.array([[40, -70, 42320, 0],
					  [42, -71, 43989, 1],
					  [38, -72, 32324, 2],
					  [42, -70, 23222, 3],
					  [37, -71, 81999, 3]])
length_example = [3, 2]
weight_example = [1/2, 1/2]

model = gmove.GroupLevelHMM(n_components=2, init_params='mce', weights=weight_example)
model.set_weights(weight_example)
model.fit(X_example, length_example)
# model._init(X_example, length_example, weight_example)

# obs = X_example[0:3]
# framelogprob = model._compute_log_likelihood(obs)
# logprob, fwdlattice = model._do_forward_pass(framelogprob)
# bwdlattice = model._do_backward_pass(framelogprob)
# posteriors = model._compute_posteriors(fwdlattice, bwdlattice)
# stats = model._initialize_sufficient_statistics()
# model._accumulate_sufficient_statistics(stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)