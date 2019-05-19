# Hidden Markov Model for GMove
# Based on hmmlearn library 
# https://hmmlearn.readthedocs.io
# 
# Author: Seth Lee 

import numpy as np
from sklearn import cluster
from sklearn.utils import check_random_state
from hmmlearn import hmm, base, _hmmc

from hmmlearn import _utils
from hmmlearn.base import _BaseHMM
from hmmlearn.utils import fill_covars, normalize, log_normalize, iter_from_X_lengths, log_mask_zero
from hmmlearn.stats import log_multivariate_normal_density

COVARIANCE_TYPES = frozenset(("spherical", "diag", "full", "tied"))

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


	def __init__(self, n_components=1, loc_covariance_type='full',
				 time_covariance_type='full',
				 loc_min_covar=1e-3,
				 time_min_covar=1e-3,

				 startprob_prior=1.0, transmat_prior=1.0,

				 loc_means_prior=0, loc_means_weight=0,
				 loc_covars_prior=1e-2, loc_covars_weight=1,
				 time_means_prior=0, time_means_weight=0,
				 time_covars_prior=1e-2, time_covars_weight=1,

				 algorithm="viterbi", random_state=None,
				 n_iter=10, tol=1e-2, verbose=False,
				 params="stmc", init_params="stmc", weights=None):

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
		self._weights = weights

		self.X_loc = []

	def set_weights(self, weights):
		self._weights = weights

	def _init(self, X, lengths=None):
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
			this HMM is representing. The sum of
			these should be ``n_samples``.
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

		if len(lengths) != len(self._weights):
			raise ValueError('Unexpected number of lengths and weights')

		# split X to 3 matrices
		self.X_loc, self.X_time, self.X_category = self._split_X_by_features(X)

		# if ``means`` is initialized
		if 'm' in self.init_params or not hasattr(self, "loc_means_"):
			# set means_ for location
			loc_kmeans = cluster.KMeans(n_clusters=self.n_components, 
									random_state=self.random_state)
			loc_kmeans.fit(self.X_loc)  # fit for lat, long
			self.loc_means_ = loc_kmeans.cluster_centers_   # loc_means_ : Mean for each states

		if 'm' in self.init_params or not hasattr(self, "time_means_"):
			# set means_ for time
			time_kmeans = cluster.KMeans(n_clusters=self.n_components, 
									random_state=self.random_state)
			time_kmeans.fit(self.X_time)
			self.time_means_ = time_kmeans.cluster_centers_ # time_means_ : Mean for each states

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

		# check weights
		if(len(self._weights) != len(lengths)):
			raise ValueError("``weights`` and ``lengths`` size mismatch")

	def _split_X_by_features(self, X):
		"""
		Split the given ```X``` according to loc, time, and category

		"""
		X_category = X[:,3:4].astype('int')
		return X[:,:2], X[:,2:3], X_category


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

		_utils._validate_covars(self._loc_covars_, self.loc_covariance_type,
								self.n_components)
		_utils._validate_covars(self._time_covars_, self.time_covariance_type,
								self.n_components)
		_check_input_symbols()

	@property
	def loc_covars_(self):
		"""Return covars as a full matrix."""
		return fill_covars(self._loc_covars_, self.loc_covariance_type,
						   self.n_components, 2)

	@loc_covars_.setter
	def loc_covars_(self, covars):
		self._loc_covars_ = np.asarray(covars).copy()

	@property
	def time_covars_(self):
		"""Return covars as a full matrix."""
		return fill_covars(self._time_covars_, self.time_covariance_type,
						   self.n_components, 2)

	@time_covars_.setter
	def time_covars_(self, covars):
		self._time_covars_ = np.asarray(covars).copy()

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
		random_state = check_random_state(random_state) # get RandomState


		# This simply samples all features individually
		# @TODO see if HMM can only outcome observations that are already seen?

		_loc_sample = random_state.multivariate_normal(
			self.loc_means_[state], self.loc_covars_[state]
		)
		_time_sample = random_state.multivariate_normal(
			self.time_means_[state], self.time_covars_[state]
		)
		cdf = np.cumsum(self.category_emissionprob_[state, :])
		_category_sample = [(cdf > random_state.rand()).argmax()]

		return np.concatenate([_loc_sample, _time_sample, _category_sample])

	def _check_input_symbols(self):
		"""Check if ``X`` is a sample for GMove

		That is ``X`` should be an array with 1st and 2nd column from bi-variate Gaussian Dist,
		3rd coulmn from uni-variate, and 4th from multinomial
		"""
		if not self._check_multinomial(self.X_category):
			raise ValueError("Expected samples from a Multinomial Distribution"
							 " for a 4th column")
	def _compute_log_likelihood(self, X):
		"""
		Here, we stupidly add log likelihoods of all features
		"""
		X_loc, X_time, X_category = self._split_X_by_features(X)

		loc_ll = log_multivariate_normal_density(
			X_loc, self.loc_means_, self.loc_covars_, self.loc_covariance_type)

		time_ll = log_multivariate_normal_density(
			X_time, self.time_means_, self.time_covars_, self.time_covariance_type)

		category_ll = np.log(self.category_emissionprob_)[:, np.concatenate(X_category)].T

		return (loc_ll + time_ll + category_ll) / 3


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

	def _initialize_sufficient_statistics(self):
		"""
		this function is *pure*, which means it doesn't change the
		state of the instance

		Returns
		-------
		nobs : int
			Number of samples in the data.

		start : array, shape (n_components, )
			An array where the i-th element corresponds to the posterior
			probability of the first sample being generated by the i-th
			state.

		trans : array, shape (n_components, n_components)
			An array where the (i, j)-th element corresponds to the
			posterior probability of transitioning between the i-th to j-th
			states.

		post : array, shape (n_components, )
			The posterior state probabilities are the conditional probabilities
			of being at state k at step i, given the observed sequence of symbols

		obs : array, shape (n_components, n_features)
			State observation likelihood

		obs**2 : array, shape (n_components, n_features)
			@TODO figure out what this is

		"""
		stats = super(GroupLevelHMM, self)._initialize_sufficient_statistics()
		stats['post'] = np.zeros(self.n_components)
		stats['loc_obs'] = np.zeros((self.n_components, 2))
		stats['loc_obs**2'] = np.zeros((self.n_components, 2))
		stats['time_obs'] = np.zeros((self.n_components, 1))
		stats['time_obs**2'] = np.zeros((self.n_components, 1))
		stats['cat_obs'] = np.zeros((self.n_components, self.n_categories))

		# @TODO figure out this part
		if self.loc_covariance_type in ('tied', 'full'):
			stats['loc_obs*obs.T'] = np.zeros((self.n_components, 2,\
										   2))
		if self.time_covariance_type in ('tied', 'full'):
			stats['time_obs*obs.T'] = np.zeros((self.n_components, 1,\
										   1))

		return stats

	def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
										  posteriors, fwdlattice, bwdlattice):
		"""
		Updates sufficient statistics from a given sample.

		Parameters
		-------
		stats : dict
			refer to _initialize_sufficient_statistics()

		obs : array, shape (length_of_sample, n_features)
			a single trajectory

		framelogprob : array, shape(, n_components)
			log likelihood of the sample trajectory

		posteriors : array, shape (, n_components)
			Posterior probabilities of each sample being generated by each
			of the model states.
		"""

		# @continue here 
		w_r = self._weights[stats['nobs']]

		# formula (1) from the paper - only the part inside sigma
		stats['nobs'] += 1  # current sample
		if 's' in self.params:
			stats['start'] += w_r * posteriors[0]

		if 't' in self.params:
			n_samples, n_components = framelogprob.shape
			# when the sample is of length 1, it contains no transitions
			# so there is no reason to update our trans. matrix estimate
			if n_samples <= 1:
				return

			# formula (2)
			log_xi_sum = np.full((n_components, n_components), -np.inf)
			_hmmc._compute_log_xi_sum(n_samples, n_components, fwdlattice,
									  log_mask_zero(self.transmat_),
									  bwdlattice, framelogprob,
									  log_xi_sum)
			log_xi_sum *= w_r
			with np.errstate(under="ignore"):
				stats['trans'] += np.exp(log_xi_sum)

		obs_loc, obs_time, obs_category = self._split_X_by_features(obs)    # @TODO valid?

		if 'm' in self.params or 'c' in self.params:
			stats['post'] += w_r * posteriors.sum(axis=0)
			stats['loc_obs'] += w_r * np.dot(posteriors.T, obs_loc)
			stats['time_obs'] += w_r * np.dot(posteriors.T, obs_time)

		if 'c' in self.params:
			if self.loc_covariance_type in ('spherical', 'diag'):
				stats['loc_obs**2'] += w_r * np.dot(posteriors.T, loc_obs ** 2)
				stats['time_obs**2'] += w_r * np.dot(posteriors.T, time_obs ** 2)
			elif self.loc_covariance_type in ('tied', 'full'):
				# posteriors: (nt, nc); obs: (nt, nf); obs: (nt, nf)
				# -> (nc, nf, nf)
				stats['loc_obs*obs.T'] += w_r * np.einsum(
					'ij,ik,il->jkl', posteriors, obs_loc, obs_loc)
				stats['time_obs*obs.T'] += w_r * np.einsum(
					'ij,ik,il->jkl', posteriors, obs_time, obs_time)

		if 'e' in self.params:
			for t, symbol in enumerate(np.concatenate(obs_category)):
				stats['cat_obs'][:, symbol] += w_r * posteriors[t]


	def _do_mstep(self, stats):
		"""Performs the M-step of EM algorithm.
		Parameters
		----------
		stats : dict
			Sufficient statistics updated from all available samples.

		Note
		----
		posteriors : array, shape (n_samples, n_components)
			Posterior probabilities of each sample being generated by each
			of the model states.
		"""

		# @TODO MULTIPLY weights in _accumulate

		# pi

		if 's' in self.params:
			startprob_ = self.startprob_prior - 1.0 + stats['start']
			self.startprob_ = np.where(self.startprob_ == 0.0,
										self.startprob_, startprob_)
			normalize(self.startprob_)

		# A matrix
		if 't' in self.params:
			transmat_ = self.transmat_prior - 1.0 + stats['trans']
			self.transmat_ = np.where(self.transmat_ == 0.0,
									  self.transmat_, transmat_)
			normalize(self.transmat_, axis=1)

		loc_means_prior = self.loc_means_prior
		loc_means_weight = self.loc_means_weight
		time_means_prior = self.time_means_prior
		time_means_weight = self.time_means_weight

		denom = stats['post'][:, np.newaxis]

		if 'm' in self.params:  # if we update means
			self.loc_means_ = ((loc_means_weight * loc_means_prior + stats['loc_obs'])
						   / (loc_means_weight + denom))
			self.time_means_ = ((time_means_weight * time_means_prior + stats['time_obs'])
						   / (time_means_weight + denom))

		if 'c' in self.params:  # if we update covars
			loc_covars_prior = self.loc_covars_prior
			loc_covars_weight = self.loc_covars_weight
			time_covars_prior = self.time_covars_prior
			time_covars_weight = self.time_covars_weight

			loc_meandiff = self.loc_means_ - loc_means_prior
			time_meandiff = self.time_means_ - time_means_prior

			if self.loc_covariance_type in ('spherical', 'diag'):
				loc_cv_num = (loc_means_weight * loc_meandiff**2
						  + stats['loc_obs**2']
						  - 2 * self.loc_means_ * stats['loc_obs']
						  + self.loc_means_**2 * denom)         # formula (4) last two parts

				loc_cv_den = max(loc_covars_weight - 1, 0) + denom

				self._loc_covars_ = \
					(loc_covars_prior + loc_cv_num) / np.maximum(loc_cv_den, 1e-5)

				if self.loc_covariance_type == 'spherical':
					self._loc_covars_ = np.tile(
						self._loc_covars_.mean(1)[:, np.newaxis],
						(1, self._loc_covars_.shape[1]))

		
			if self.time_covariance_type in ('spherical', 'diag'):
				time_cv_num = (time_means_weight * time_meandiff**2
						  + stats['time_obs**2']
						  - 2 * self.time_means_ * stats['time_obs']
						  + self.time_means_**2 * denom)

				time_cv_den = max(time_covars_weight - 1, 0) + denom

				self._time_covars_ = \
					(time_covars_prior + time_cv_num) / np.maximum(time_cv_den, 1e-5)

				if self.time_covariance_type == 'spherical':
					self._time_covars_ = np.tile(
						self._time_covars_.mean(1)[:, np.newaxis],
						(1, self._time_covars_.shape[1]))

			if self.loc_covariance_type in ('tied', 'full'):
				cv_num = np.empty((self.n_components, 2,
				                2)) 	# n_features for gaussianHMM : 2
				for c in range(self.n_components):
				  obsmean = np.outer(stats['loc_obs'][c], self.loc_means_[c])

				  cv_num[c] = (loc_means_weight * np.outer(loc_meandiff[c],
				                                       loc_meandiff[c])
				               + stats['loc_obs*obs.T'][c]
				               - obsmean - obsmean.T
				               + np.outer(self.loc_means_[c], self.loc_means_[c])
				               * stats['post'][c])
				cvweight = max(loc_covars_weight - 2, 0)
				if self.loc_covariance_type == 'tied':
				  self._loc_covars_ = ((loc_covars_prior + cv_num.sum(axis=0)) /
				                   (cvweight + stats['post'].sum()))
				elif self.loc_covariance_type == 'full':
				  self._loc_covars_ = ((loc_covars_prior + cv_num) /
				                   (cvweight + stats['post'][:, None, None]))

			if self.time_covariance_type in ('tied', 'full'):
				cv_num = np.empty((self.n_components, 1,
				                1)) 	# n_features for gaussianHMM : 2
				for c in range(self.n_components):
				  obsmean = np.outer(stats['time_obs'][c], self.time_means_[c])

				  cv_num[c] = (time_means_weight * np.outer(time_meandiff[c],
				                                       time_meandiff[c])
				               + stats['time_obs*obs.T'][c]
				               - obsmean - obsmean.T
				               + np.outer(self.time_means_[c], self.time_means_[c])
				               * stats['post'][c])
				cvweight = max(time_covars_weight - 1, 0)
				if self.time_covariance_type == 'tied':
				  self._time_covars_ = ((time_covars_prior + cv_num.sum(axis=0)) /
				                   (cvweight + stats['post'].sum()))
				elif self.time_covariance_type == 'full':
				  self._time_covars_ = ((time_covars_prior + cv_num) /
				                   (cvweight + stats['post'][:, None, None]))

			if 'e' in self.params:
				self.emissionprob_ = (stats['cat_obs']
								  / stats['cat_obs'].sum(axis=1)[:, np.newaxis])
