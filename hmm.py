# Hidden Markov Model for GMove
# Based on hmmlearn library 
# https://hmmlearn.readthedocs.io
# 
# Author: Seth Lee 

import numpy as np
from sklearn import cluster
from sklearn.utils import check_random_state
from hmmlearn import hmm, base

from hmmlearn.base import _BaseHMM
from hmmlearn import _utils

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
	category_emissionprob_ : array, shape (n_components, n_categories)
		emission probability for categories

	n_categories : int
		number of venue categories

	"""

	def __init__(self, n_components=1, loc_covariance_type='diag',
				 time_covariance_type='diag',
				 loc_min_covar=1e-3,
				 time_min_covar=1e-3,

				 startprob_prior=1.0, transmat_prior=1.0,

				 loc_means_prior=0, loc_means_weight=0,
				 loc_covars_prior=1e-2, loc_covars_weight=1,
				 time_means_prior=0, time_means_weight=0,
				 time_covars_prior=1e-2, time_covars_weight=1,

				 algorithm="viterbi", random_state=None,
				 n_iter=10, tol=1e-2, verbose=False,
				 params="stmc", init_params="stmc"):

		_BaseHMM.__init__(self, n_components,
						  startprob_prior=startprob_prior,
						  transmat_prior=transmat_prior, algorithm=algorithm,
						  random_state=random_state, n_iter=n_iter,
						  tol=tol, params=params, verbose=verbose,
						  init_params=init_params)

		self.loc_covariance_type = loc_covariance_type 
		self.time_covariance_type = time_covariance_type 
		self.loc_min_covar = loc_min_covar
		self.time_min_covar = time_min_covar
		self.loc_means_prior = loc_means_prior
		self.time_means_prior = time_means_prior
		self.loc_means_weight = loc_means_weight
		self.time_means_weight = time_means_weight
		self.loc_covars_prior = loc_covars_prior
		self.time_covars_prior = time_covars_prior
		self.loc_covars_weight = loc_covars_weight
		self.time_covars_weight = time_covars_weight

		self.X_loc = []

	def _init(self, X, lengths=None, weights=None):
		"""
		Initializes model parameters prior to fitting.
		
		X : array-like, shape (n_samples, n_features)
			Feature matrix of individual samples.
			n_features should be equal to 4 (lat, long, time, category).
			@TODO Implement for general cases

		lengths : array-like of integers, shape (n_sequences, )
			Lengths of the individual sequences in ``X``. The sum of
			these should be ``n_samples``.

		weights : array-like, the probability that user for 
			individual sequences in ``X`` belongs to the group
			this HMM is representing
		"""
		
		super(GroupLevelHMM, self)._init(X, lengths=lengths)

		# Check the number of features
		_, n_features = X.shape
		if hasattr(self, 'n_features') and self.n_features != n_features:
			raise ValueError('Unexpected number of dimensions, got %s but '
							 'expected %s' % (n_features, self.n_features))
		if n_features != 4:
			raise ValueError('Unexpected number of features, got %s but '
							 'expected 4' % (n_features))

		self.n_features = n_features

		if len(lengths) != len(weights):
			raise ValueError('Unexpected number of lengths and weights')

		# split X to 3 matrices
		self.X_loc = X[:,:2]
		self.X_time = X[:,2:3]
		self.X_category = X[:,3:4]

		# if ``means`` is initialized
		if 'm' in self.init_params or not hasattr(self, "loc_means_"):
			# set means_ for location
			loc_kmeans = cluster.KMeans(n_clusters=self.n_components, 
									random_state=self.random_state)
			loc_kmeans.fit(self.X_loc)	# fit for lat, long
			self.loc_means_ = loc_kmeans.cluster_centers_	# loc_means_ : Mean for each states

		if 'm' in self.init_params or not hasattr(self, "time_means_"):
			# set means_ for time
			time_kmeans = cluster.KMeans(n_clusters=self.n_components, 
									random_state=self.random_state)
			time_kmeans.fit(self.X_time)
			self.time_means_ = time_kmeans.cluster_centers_	# time_means_ : Mean for each states

		if 'c' in self.init_params or not hasattr(self, "loc_covars_"):
			cv_loc = np.cov(self.X_loc.T) + self.loc_min_covar * np.eye(self.X_loc.shape[1])
			if not cv_loc.shape:
				cv_loc.shape = (1, 1)
			self._loc_covars_ = \
				_utils.distribute_covar_matrix_to_match_covariance_type(
					cv_loc, self.loc_covariance_type, self.n_components).copy()

		if 'c' in self.init_params or not hasattr(self, "time_covars_"):
			cv_time = np.cov(self.X_time.T) + self.time_min_covar * np.eye(self.X_time.shape[1])
			if not cv_time.shape:
				cv_time.shape = (1, 1)
			self._time_covars_ = \
				_utils.distribute_covar_matrix_to_match_covariance_type(
					cv_time, self.time_covariance_type, self.n_components).copy()

		self._check_input_symbols()	# check if category column in ``X`` follows
									# multinomial distribution
		self.random_state = check_random_state(self.random_state)

		if 'e' in self.init_params:
			if not hasattr(self, "n_categories"):
				symbols = set()
				for i, j in iter_from_X_lengths(self.X_category, lengths):
					symbols |= set(self.X_category[i:j].flatten())
				self.n_categories = len(symbols)
			self.category_emissionprob_ = self.random_state \
				.rand(self.n_components, self.n_categories)
			normalize(self.category_emissionprob_, axis=1)


	def _check(self):
		"""
		just a sanity check for the HMM parameters

		"""
		super(GroupLevelHMM, self)._check()

		self.loc_means_ = np.asarray(self.loc_means_)
		self.time_means_ = np.asarray(self.time_means_)

		if self.loc_covariance_type not in COVARIANCE_TYPES:
			raise ValueError('covariance_type must be one of {}'
							 .format(COVARIANCE_TYPES))
		if self.time_covariance_type not in COVARIANCE_TYPES:
			raise ValueError('covariance_type must be one of {}'
							 .format(COVARIANCE_TYPES))

		_utils._validate_covars(self.loc_covars_, self.loc_covariance_type,
								self.n_components)
		_utils._validate_covars(self.time_covars_, self.time_covariance_type,
								self.n_components)

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

		raise NotImplementedError('This function has to be implemented to generate samples')

	def _check_input_symbols(self):
		"""Check if ``X`` is a sample for GMove

		That is ``X`` should be an array with 1st and 2nd column from bi-variate Gaussian Dist,
		3rd coulmn from uni-variate, and 4th from multinomial
		"""
		if not self._check_multinomial(self.X_category):
			raise ValueError("Expected a sample from a Multinomial Distribution"
							 " for a 4th column")
	def _compute_log_likelihood(self, X):
		loc_ll = log_multivariate_normal_density(
			X, self.loc_means_, self._loc_covars_, self.loc_covariance_type)
		time_ll = log_multivariate_normal_density(
			X, self.time_means_, self._time_covars_, self.time_covariance_type)
		category_ll = np.log(self.category_emissionprob_)[:, np.concatenate(X)].T


	def _check_multinomial(self, X):
		"""Check if ``X`` is a sample from a Multinomial Distribution
		"""
		symbols = np.concatenate(X)
		if (len(symbols) == 1                                # not enough data
			or not np.issubdtype(symbols.dtype, np.integer)  # not an integer
			or (symbols < 0).any()):                         # not positive
			return False
		u = np.unique(symbols)
		return u[0] == 0 and u[-1] == len(u) - 1




