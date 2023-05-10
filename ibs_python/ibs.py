class EstimateResult(dict):
    """Represents the result of the estimation procedure."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        order_keys = [
            "nlogl",
            "nlogl_var",
            "nlogl_std",
            "exit_flag",
            "message",
            "elapsed_time",
            "num_samples_per_trial",
            "fun_count",
        ]

        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            # Custom ordering logic
            items = [(k, self[k]) for k in order_keys if k in self]
            return "\n".join([k.rjust(m) + ": " + repr(v) for k, v in items])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


import time

import numpy as np
from scipy.special import polygamma, psi


class IBS:

    """
    IBS class for computing the negative log-likelihood of a simulator based model.
    """

    def __init__(
        self,
        fun,
        resp_mat,
        design_mat,
        vectorized=None,
        acceleration=1.5,
        num_samples_per_call=0,
        max_iter=1e5,
        max_time=np.inf,
        max_samples=1e4,
        acceleration_threshold=0.1,
        vectorized_threshold=0.1,
        max_mem=1e6,
        negLogL_threshold=np.inf,
    ):
        """
        Initializes an Inverse Binomial sampling (IBS) object with the specified parameters.

        Args:
            fun: A callable object that simulates the model's responses. fun takes as input a vector of parameters params and an experimental
            design matrix dmat (one row per trial), and generates a matrix of simulated model responses (one row per trial, corresponding to rows of dmat).
            resp_mat: A numpy array representing the observed data
            design_mat: A numpy array representing the design matrix used to calculate the function values
            vectorized: A boolean indicating whether to use a vectorized sampling algorithm with acceleration
            acceleration: The acceleration factor for vectorized sampling
            num_samples_per_call: The number of starting samples per trial per function call (0 = choose automatically)
            max_iter: The maximum number of iterations (per trial and estimate)
            max_time: The maximum time for an IBS call (in seconds)
            max_samples: The maximum number of samples per function call
            acceleration_threshold: The threshold at which to stop accelerating (in seconds)
            vectorized_threshold: The maximum threshold for using the vectorized algorithm (in seconds)
            max_mem: The maximum number of samples for the vectorized implementation
            negLogL_threshold: The threshold for the negative log-likelihood (works differently in vectorized version)
        """
        self.fun = fun
        self.resp_mat = np.atleast_1d(resp_mat)
        self.design_mat = design_mat
        self.vectorized = (
            vectorized  # Use vectorized sampling algorithm with acceleration
        )
        self.acceleration = acceleration  # Acceleration factor for vectorized sampling
        self.num_samples_per_call = num_samples_per_call  # Number of starting samples per trial per function call (0 = choose automatically)
        self.max_iter = (
            max_iter  # Maximum number of iterations (per trial and estimate)
        )
        self.max_time = max_time  # Maximum time for a IBS call (in seconds)
        self.max_samples = max_samples  # maximum number of samples per function call
        self.acceleration_threshold = acceleration_threshold  # keep accelerating until this threshold is reached (in s)
        self.vectorized_threshold = vectorized_threshold  # maximum threshold for using vectorized algorithm (in s)
        self.max_mem = (
            max_mem  # maximum number of samples for vectorized implementation
        )
        self.negLogL_threshold = (
            negLogL_threshold  # threshold for negative log-likelihood
        )

    def __call__(
        self,
        params,
        num_reps=10,
        trial_weights=None,
        additional_output=None,
        return_positive=False,
    ):
        """
        Compute the negative log-likelihood of a simulator based model.

        Input:
        params - parameter vector
        num_reps - number of repetitions (default = 10)
        trial_weights - weights vector (default = None)
        additional_output - None = return estimate (default)
                            'var' = return estimate, variance
                            'std' = return estimate, standard deviation
                            'full' = return dictionary type output:
                                ['nlogl', 'nlogl_var', 'exit_flag', 'message', 'elapsed_time','num_samples_per_trial', 'fun_count']
        return_positive - return positive log-likelihood (default = False)

        Output:
        nlogl - negative log-likelihood (if return_positive is False else positive log-likelihood)
        nlogl_var - variance of negative log-likelihood estimate (if additional_output is 'var')
        nlogl_std - standard deviation of negative log-likelihood estimate (if additional_output is 'std')
        nlogl_dict - dictionary type output (if additional_output is 'full')
        exit_flag - exit flag (0 = correct termination, 1 = negative log-likelihood threshold reached, 2 = maximum runtime reached, 3 = maximum iterations reached)
        message - exit message
        elapsed_time - elapsed time (in seconds)
        num_samples_per_trial - number of samples per trial
        fun_count - number of function evaluations


        """
        t0 = time.perf_counter()
        num_trials = self.resp_mat.shape[0]

        # weights vector should be a scalar or same length as number of trials
        weights = 1.0
        if trial_weights is not None:
            weights = trial_weights.reshape(-1)
        if not np.isscalar(weights) and len(weights) != num_trials:
            raise ValueError(
                "IBS:SizeMismatch",
                "Length of trial_weights must match the number of trials",
            )

        def compute_logl(self, params, num_reps, weights, return_positive, t0):

            sim_data = None
            elapsed_time = 0
            num_reps = int(num_reps)

            # check if vectorized or loop version should be used
            if self.vectorized is None:
                start = time.time()
                if self.design_mat is None:
                    sim_data = self.fun(params, np.arange(num_trials))
                else:
                    sim_data = self.fun(params, self.design_mat)
                elapsed_time = time.time() - start
                vectorized_flag = elapsed_time < self.vectorized_threshold
            else:
                vectorized_flag = self.vectorized

            def get_LL_from_K(psi_tab, K_mat):
                """
                Convert matrix of K values into log-likelihoods.
                """
                K_max = max(1, np.max(K_mat))
                if K_max > len(psi_tab):  # fill digamma function table
                    psi_tab = np.concatenate(
                        (psi_tab, psi(1) - psi(np.arange(len(psi_tab) + 1, K_max + 1)))
                    )
                LL_mat = psi_tab[np.maximum(1, K_mat.astype(int)) - 1]
                return LL_mat, psi_tab

            def vectorized_ibs_sampling(params, sim_data0, elapsed_time0, t0, num_reps):
                """
                A function to perform vectorized Importance-Biased Sampling (IBS) for the given parameters.

                Args:
                params (array): The model parameters.
                sim_data0 (array): The initial simulation data.
                elapsed_time0 (float): The initial elapsed time.
                t0 (float): The starting time of the IBS sampling.
                num_reps (int): The number of repetitions for each trial.

                Returns:
                nlogl (float): The negative log-likelihood.
                K (array): A matrix of samples-to-hit for each trial and repeat.
                num_reps_per_trial (array): The number of repetitions for each trial.
                num_samples_total (int): The total number of samples drawn.
                fun_count (int): The number of times the objective function was called.
                exitflag (int): The exit flag indicating the termination condition.
                """

                num_trials = self.resp_mat.shape[0]
                trials = np.arange(num_trials)  # enumerate the trials
                num_samples_total = 0  # total number of samples drawn
                fun_count = 0
                Psi_tab = []
                exit_flag = 0

                # Empty matrix of K values (samples-to-hit) for each repeat for each trial
                K_mat = np.zeros((num_reps, num_trials), dtype=int)

                # Matrix of rep counts
                K_place0 = np.tile(np.arange(num_reps)[:, np.newaxis], (1, num_trials))

                # Current rep being sampled for each trial
                Ridx = np.zeros(num_trials)

                # Current vector of "open" K values per trial (not reached a "hit" yet)
                K_open = np.zeros(num_trials)

                target_hits = num_reps * np.ones(num_trials)
                max_iter = int(self.max_iter * num_reps)

                if self.num_samples_per_call == 0:
                    samples_level = num_reps
                else:
                    samples_level = self.num_samples_per_call

                for iter in range(max_iter):
                    # Pick trials that need more hits, sample multiple times
                    T = trials[Ridx < target_hits]

                    # Check if max time has been reached
                    if (
                        np.isfinite(self.max_time)
                        and time.perf_counter() - t0 > self.max_time
                    ):
                        T = np.empty(0)
                        exit_flag = 2
                        try:
                            raise RuntimeWarning(
                                "Warning in IBS execution: termination after maximum execution time was reached (the estimate can be arbitrarily biased)"
                            )
                        except RuntimeWarning as e:
                            print(e)

                    if len(T) == 0:
                        break
                    num_considered_trials = len(T)
                    # With accelerated sampling, might request multiple samples at once
                    num_samples = min(max(1, np.round(samples_level)), self.max_samples)
                    max_samples = np.ceil(self.max_mem / num_considered_trials)
                    num_samples = min(num_samples, max_samples)
                    Tmat = np.tile(T, (int(num_samples), 1))

                    # Simulate trials
                    if iter == 0 and num_samples == 1 and sim_data0 is not None:
                        sim_data = sim_data0
                        elapsed_time = elapsed_time0
                    else:
                        start = time.time()
                        if self.design_mat is None:
                            sim_data = self.fun(params, Tmat.reshape(-1))
                        else:
                            sim_data = self.fun(
                                params, self.design_mat[Tmat.reshape(-1)]
                            )
                        fun_count += 1
                        elapsed_time = time.time() - start

                    # Check that the returned simulated data have the right size
                    if len(sim_data) != np.size(Tmat):
                        raise ValueError(
                            "IBS: number of rows of returned simulated data does not match the number of requested trials"
                        )

                    num_samples_total += num_considered_trials

                    # Accelerated sampling
                    if (
                        self.acceleration > 0
                        and elapsed_time < self.acceleration_threshold
                    ):
                        samples_level = samples_level * self.acceleration

                    # Check for hits
                    hits_temp = self.resp_mat[Tmat.reshape(-1)] == sim_data

                    def get_K_from_hits(hits_temp):

                        # Build matrix of new hits (sandwich with buffer of hits, then removed)
                        hits_new = np.concatenate(
                            (
                                np.ones((1, num_considered_trials)),
                                hits_temp.reshape(Tmat.shape),
                                np.ones((1, num_considered_trials)),
                            ),
                            axis=0,
                        )

                        # Extract matrix of Ks from matrix of hits for this iteration
                        list = np.nonzero(hits_new.T)
                        row = list[0]
                        delta = np.diff(np.append(list[1], 0))
                        remove_idx = delta <= 0
                        row = row[~remove_idx]
                        delta = delta[~remove_idx]
                        index_col = np.nonzero(
                            np.diff(np.concatenate((np.array([-1]), row)))
                        )
                        col = np.arange(len(row)) - np.take(index_col, row)
                        K_iter = np.zeros((len(T), np.max(col) + 1))
                        K_iter[row, col] = delta
                        return K_iter

                    K_iter = get_K_from_hits(hits_temp)

                    # Add K_open to first column of K_iter
                    K_iter[:, 0] = K_iter[:, 0] + K_open[T]

                    # Find last K position for each trial
                    index_last = (
                        np.argmin(
                            np.hstack(
                                (K_iter, np.zeros(num_considered_trials).reshape(-1, 1))
                            ),
                            axis=1,
                        )
                        - 1
                    )
                    row_index = np.arange(len(T))
                    # Subtract one hit from last K (it was added)
                    K_iter[row_index, index_last] = K_iter[row_index, index_last] - 1
                    K_open[T] = K_iter[row_index, index_last]

                    # For each trial, ignore entries of K_iter past max number of reps
                    index_mat = (
                        np.tile(
                            np.arange(K_iter.shape[1])[:, np.newaxis],
                            (1, num_considered_trials),
                        )
                        + Ridx[T]
                    )
                    K_iter[index_mat.T >= num_reps] = 0

                    # Find last K position for each trial again
                    index_last2 = (
                        np.argmin(
                            np.hstack(
                                (K_iter, np.zeros(num_considered_trials).reshape(-1, 1))
                            ),
                            axis=1,
                        )
                        - 1
                    )

                    # Add current K to full K matrix
                    K_iter_place = (K_place0[:, :num_considered_trials] >= Ridx[T]) & (
                        K_place0[:, :num_considered_trials] <= Ridx[T] + index_last2
                    )
                    K_place = np.zeros_like(K_place0, dtype=bool)
                    K_place[:, T] = K_iter_place
                    K_mat_flat = K_mat.flatten("F")
                    K_mat_flat[K_place.flatten("F")] = K_iter[K_iter > 0].flatten()
                    K_mat = K_mat_flat.reshape(K_mat.shape, order="F")
                    Ridx[T] = Ridx[T] + index_last

                    # Compute log-likelihood only if a threshold is set
                    if np.isfinite(self.negLogL_threshold):
                        Rmin = np.min(Ridx[T])
                        if Rmin >= K_mat.shape[0]:
                            continue
                        LL_temp, Psi_tab = get_LL_from_K(Psi_tab, K_mat[int(Rmin), :])
                        nLL_temp = -np.sum(LL_temp, axis=0)
                        if nLL_temp > self.negLogL_threshold:
                            index_move = Ridx == Rmin
                            Ridx[index_move] = Rmin + 1
                            K_open[index_move] = 0
                            exit_flag = 1

                else:
                    exit_flag = 3
                    try:
                        raise RuntimeWarning(
                            "Warning in IBS execution: termination after maximum number of iterations was reached (the estimate can be arbitrarily biased)"
                        )
                    except RuntimeWarning as e:
                        print(e)

                if np.isfinite(self.negLogL_threshold) and exit_flag == 1:
                    try:
                        raise RuntimeWarning(
                            "Warning in IBS execution: termination after negative log-likelihood threshold was reached (the estimate is biased)"
                        )
                    except RuntimeWarning as e:
                        print(e)

                # Compute log-likelihood
                num_reps_per_trial = np.sum(
                    K_mat > 0, axis=0
                )  # number of repetitions of the single trials
                LL_mat, Psi_tab = get_LL_from_K(Psi_tab, K_mat)
                nlogl = -np.sum(LL_mat, axis=0) / num_reps_per_trial
                K = K_mat.T

                return (
                    nlogl,
                    K,
                    num_reps_per_trial,
                    num_samples_total,
                    fun_count,
                    exit_flag,
                )

            def loop_ibs_sampling(params, sim_data0, t0, num_reps):
                """
                A function to perform loop-based Importance-Biased Sampling (IBS) for the given parameters.

                Args:
                params (array): The model parameters.
                sim_data0 (array): The initial simulation data.
                t0 (float): The starting time of the IBS sampling.
                num_reps (int): The number of repetitions for each trial.

                Returns:
                nlogl (float): The negative log-likelihood.
                K (array): A matrix of samples-to-hit for each trial and repeat.
                num_reps_per_trail (array): The number of repetitions for each trial.
                num_samples_total (int): The total number of samples drawn.
                fun_count (int): The number of times the objective function was called.
                exitflag (int): The exit flag indicating the termination condition.
                """

                num_trials = self.resp_mat.shape[0]

                trials = np.arange(num_trials)  # enumerate the trials
                max_iter = self.max_iter

                K = np.zeros(
                    (num_trials, num_reps)
                )  # saves the number of iterations needed for the sample to match the trial response
                num_samples_total = 0  # total number of samples drawn
                fun_count = 0
                psi_tab = []
                exit_flag = 0

                for iRep in range(num_reps):

                    if exit_flag == 2:
                        break
                    if (
                        np.isfinite(self.max_time)
                        and time.perf_counter() - t0 > self.max_time
                    ):
                        exit_flag = 2
                        try:
                            raise RuntimeWarning(
                                "Warning in IBS execution: termination after maximum execution time was reached (the estimate can be arbitrarily biased)"
                            )
                        except RuntimeWarning as e:
                            print(e)
                        break

                    offset = 1
                    hits = np.zeros(num_trials, dtype=bool)

                    for iter in range(max_iter):

                        T = trials[hits == False]
                        if len(T) == 0:
                            break
                        if iter == 0 and iRep == 0 and sim_data0 is not None:
                            sim_data = sim_data0
                            fun_count += 1
                        elif self.design_mat is None:
                            # call function with input params only for the trials that have not been hit yet
                            sim_data = self.fun(params, T)
                        else:
                            # call function with input params and design_mat only for the trials that have not been hit yet
                            sim_data = self.fun(params, self.design_mat[T])
                            fun_count += 1

                        if np.shape(np.atleast_1d(sim_data))[0] != len(T):
                            raise ValueError(
                                "IBS: number of rows of returned simulated data does not match the number of requested trials"
                            )
                        num_samples_total += len(T)
                        hits_new = sim_data == self.resp_mat[T]
                        hits[T] = hits_new

                        K[np.atleast_1d(T)[hits_new], iRep] = offset
                        offset += 1

                        if np.isfinite(self.negLogL_threshold):
                            K[hits == False, iRep] = offset
                            LL_mat, psi_tab = get_LL_from_K(psi_tab, K[:, iRep])
                            nlogl = -np.sum(
                                LL_mat, axis=0
                            )  # compute the negative log-likelihood of the current repetition
                            if nlogl > self.negLogL_threshold:
                                T = []
                                exit_flag = 1
                                break
                        # Terminate if above maximum allowed runtime
                        if (
                            np.isfinite(self.max_time)
                            and time.perf_counter() - t0 > self.max_time
                        ):
                            T = []
                            exit_flag = 2
                            try:
                                raise RuntimeWarning(
                                    "Warning in IBS execution: termination after maximum execution time was reached (the estimate can be arbitrarily biased)"
                                )
                            except RuntimeWarning as e:
                                print(e)
                            break

                    else:
                        exit_flag = 3
                        try:
                            raise RuntimeWarning(
                                "Warning in IBS execution: termination after maximum number of iterations was reached (the estimate can be arbitrarily biased)"
                            )
                        except RuntimeWarning as e:
                            print(e)

                if exit_flag == 1:
                    try:
                        raise RuntimeWarning(
                            "Warning in IBS execution: termination after negative log-likelihood threshold was reached (the estimate is biased)"
                        )
                    except RuntimeWarning as e:
                        print(e)

                num_reps_per_trail = np.sum(
                    K > 0, axis=1
                )  # number of repetitions of the single trials
                LL_mat, psi_tab = get_LL_from_K(psi_tab, K)
                nlogl = -np.sum(LL_mat, axis=1) / num_reps_per_trail

                return (
                    nlogl,
                    K,
                    num_reps_per_trail,
                    num_samples_total,
                    fun_count,
                    exit_flag,
                )

            if vectorized_flag:
                (
                    nlogl,
                    K,
                    num_reps_per_trial,
                    num_samples_total,
                    fun_count,
                    exit_flag,
                ) = vectorized_ibs_sampling(
                    params, sim_data, elapsed_time, t0, num_reps
                )
            else:
                (
                    nlogl,
                    K,
                    num_reps_per_trial,
                    num_samples_total,
                    fun_count,
                    exit_flag,
                ) = loop_ibs_sampling(params, sim_data, t0, num_reps)

            nlogl = np.sum(nlogl * weights)
            if return_positive:
                nlogl = -nlogl
            return nlogl, K, num_reps_per_trial, num_samples_total, fun_count, exit_flag

        (
            nlogl,
            K,
            num_reps_per_trial,
            num_samples_total,
            fun_count,
            exit_flag,
        ) = compute_logl(self, params, num_reps, weights, return_positive, t0)

        if additional_output in [None, "none"]:
            return nlogl
        elif additional_output not in [None, "none"]:
            # compute variance of log-likelihood
            K_max = np.amax(K, initial=1)
            K_tab = -polygamma(1, np.arange(1, K_max + 1)) + polygamma(1, 1)
            LLvar = K_tab[np.maximum(1, K.astype(int)) - 1]
            nlogl_var = np.sum(LLvar, axis=1) / num_reps_per_trial**2
            nlogl_var = np.sum(nlogl_var * (weights**2))
            if additional_output == "var":
                return nlogl, nlogl_var
            if additional_output == "std":
                return nlogl, np.sqrt(nlogl_var)
            if additional_output == "full":
                message = ""
                if exit_flag == 0:
                    message = "correct termination (the estimate is unbiased)"
                elif exit_flag == 1:
                    message = "termination after negative log-likelihood threshold was reached (the estimate is biased)"
                elif exit_flag == 2:
                    message = "termination after maximum execution time was reached (the estimate can be arbitrarily biased)"
                elif exit_flag == 3:
                    message = "termination after maximum number of iterations was reached (the estimate can be arbitrarily biased)"

                return EstimateResult(
                    nlogl=nlogl,
                    nlogl_var=nlogl_var,
                    nlogl_std=np.sqrt(nlogl_var),
                    exit_flag=exit_flag,
                    message=message,
                    elapsed_time=time.perf_counter() - t0,
                    num_samples_per_trial=num_samples_total / num_trials,
                    fun_count=fun_count,
                )
        else:
            raise ValueError(
                "IBS:InvalidArgument", "Invalid value for additional_output."
            )
