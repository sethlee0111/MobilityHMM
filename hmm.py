# Hidden Markov Model for GMove
# Based on hmmlearn library 
# https://hmmlearn.readthedocs.io
# 
# Author: Seth Lee 

import numpy as np
from hmmlearn import hmm, base

class GroupLevelHMM(_BaseHMM):
	"""Hidden Markov Model with emmision probability for GMove

	Parameters
    ----------
    n_components : int
        Number of states in the model.

	loc_params : dict
		Parameters for location observations
	
	time_params : dict
		Parameters for time observations

	keywords_params : dict
		Parameters for keywords observations

	Attributes
    ----------
	_loc_HMM : HMM instance for managing variable for emission probabilities for location
	_time_HMM : HMM instance for managing variable for emission probabilities for time
	_category_HMM : HMM instance for managing variable for emission probabilities for category
	"""
	
	def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stmc", init_params="stmc",
                 loc_params, time_params, category_params):

        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params)

    	self._loc_HMM = GaussianHMM(loc_params)
    	self._time_HMM = GaussianHMM(time_params)
    	self._category_HMM = MultinomialHMM(category_params)

	def _init(self, X, lengths=None):
		"""
		Initializes model parameters prior to fitting.
		
		X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.
        """
        self._loc_HMM._init(X, lengths)
        self._time_HMM._init(X, lengths)
        self._category_HMM._init(X, lengths)

    def _check(self):
    	"""
    	sanity check for the HMM parameters

    	"""
    	self._loc_HMM._check()
        self._time_HMM._check()
        self._category_HMM._check()

    def _generate_sample_from_state(self, state, random_state):
    	"""Generate random samples from the model.
		( Generate observations from the state )
        Parameters
        ----------
        state : int
            Index of the component to condition on
        random_state : RandomState or an int seed
            A random number generator instance. If ``None``, the object's
            ``random_state`` is used.
        Returns
        -------
        X : array, shape (n_samples, n_features)
            Feature matrix.
        state_sequence : array, shape (n_samples, )
            State sequence produced by the model.
        """

        # check_random_state() : return a new RandomState instance seeded with seed
        random_state = self.check_random_state(random_state)	# get RandomState


        _category_n = random_state.multivariate_normal(
            self.means_[state], self.covars_[state]
        )
        
        
        # loc_random_state = self._loc_HMM.check_random_state(_loc_HMM.random_state)
        # time_random_state = self._time_HMM.check_random_state(_time_HMM.random_state)
        # category_random_state = self._keywords_HMM.check_random_state(_category_HMM.random_state)








