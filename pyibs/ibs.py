class EstimateResult(dict):
    """
    Dictionary type to represent the result of the estimation procedure with additional information.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        order_keys = [
            "neg_logl",
            "neg_logl_var",
            "neg_logl_std",
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

    Parameters
    ----------
    sample_from_model: callable
        Simulates the model's responses. It takes as input a vector of parameters params and design_matrix
        and generates a matrix of simulated model responses (one row per trial, corresponding to rows of design_matrix).
    response_matrix: np.array
        The observed responses.
    design_matrix: np.array
        The design matrix used as input to sample from the model.
    vectorized: boolean, optional
        Indicates whether to use a vectorized sampling algorithm with acceleration, default = None.
        If None, the vectorized algorithm is used if the time to generate samples for each trial is less than vectorized_threshold.
    acceleration: float, optional
        The acceleration factor for vectorized sampling, default = 1.5.
    num_samples_per_call: int, optional
        The number of starting samples per trial per function call.
        If equal to 0 the number of starting samples is chosen automatically, default = 0.
    max_iter: int, optional
        The maximum number of iterations (per trial and estimate), default = 1e5.
    max_time: float, optional
        The maximum time for an IBS call (in seconds), default = np.inf.
    max_samples: int, optional
        The maximum number of samples per function call, default = 1e4.
    acceleration_threshold: float, optional
        The threshold at which to stop accelerating (in seconds), default = 0.1.
    vectorized_threshold: float, optional
        The maximum threshold for using the vectorized algorithm (in seconds), default = 0.1.
    max_mem: int, optional
        The maximum number of samples for the vectorized implementation, default = 1e6.
    neg_logl_threshold: float, optional
        The threshold for the negative log-likelihood (works differently in vectorized version), default = np.inf.

    """

    def __init__(
        self,
        sample_from_model,
        response_matrix,
        design_matrix,
        vectorized=None,
        acceleration=1.5,
        num_samples_per_call=0,
        max_iter=10 ^ 5,
        max_time=np.inf,
        max_samples=1e4,
        acceleration_threshold=0.1,
        vectorized_threshold=0.1,
        max_mem=1e6,
        neg_logl_threshold=np.inf,
    ):
        self.sample_from_model = sample_from_model
        self.response_matrix = np.atleast_1d(response_matrix)
        self.design_matrix = design_matrix
        self.vectorized = vectorized
        self.acceleration = acceleration
        self.num_samples_per_call = num_samples_per_call
        self.max_iter = max_iter
        self.max_time = max_time
        self.max_samples = max_samples
        self.acceleration_threshold = acceleration_threshold
        self.vectorized_threshold = vectorized_threshold
        self.max_mem = max_mem
        self.neg_logl_threshold = neg_logl_threshold

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

        Parameters
        ----------
        params: np.array
            The parameter vector.
        num_reps: int, optional
            The number of repetitions, default = 10.
        trial_weights: np.array, optional
            The trial weights vector, default = None.
        additional_output: str, optional
            The output type, if equal to None then only the negative log-likelihood is returned, default = None.
            If equal to 'var' then the negative log-likelihood and the variance of the negative log-likelihood estimate is returned.
            If equal to 'std' then the negative log-likelihood and the standard deviation of the negative log-likelihood estimate is returned.
            If equal to 'full' then a dictionary type output is returned with following additional information about the sampling:
            exit_flag - The exit flag (0 = correct termination, 1 = negative log-likelihood threshold reached, 2 = maximum runtime reached, 3 = maximum iterations reached).
            message - The exit message.
            elapsed_time - The elapsed time (in seconds).
            num_samples_per_trial - The number of samples per trial.
            fun_count - The number of time the sample_from_model function was called.
        return_positive: boolean, optional
            Indicates whether to return the positive log-likelihood, default = False.

        Returns
        ----------
        neg_logl: float
            The negative log-likelihood (if return_positive is False else positive log-likelihood).
        neg_logl_var: float
            The variance of the negative log-likelihood estimate (if additional_output is 'var').
        neg_logl_std: float
            The standard deviation of the negative log-likelihood estimate (if additional_output is 'std').
        If additional_output is 'full' then a dictionary type output is returned with following additional information about the sampling:
        exit_flag: int
            The exit flag (0 = correct termination, 1 = negative log-likelihood threshold reached, 2 = maximum runtime reached, 3 = maximum iterations reached).
        message: str
            The exit message.
        elapsed_time: float
            The elapsed time (in seconds).
        num_samples_per_trial: int
            The number of samples per trial.
        fun_count: int
            The number of sample_from_model function evaluations in the call.
        """
        t0 = time.perf_counter()
        num_trials = self.response_matrix.shape[0]

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

            simulated_data = None
            elapsed_time = 0
            num_reps = int(num_reps)

            # check if vectorized or loop version should be used
            if self.vectorized is None:
                start = time.time()
                if self.design_matrix is None:
                    simulated_data = self.sample_from_model(
                        params, np.arange(num_trials)
                    )
                else:
                    simulated_data = self.sample_from_model(
                        params, self.design_matrix
                    )
                elapsed_time = time.time() - start
                vectorized_flag = elapsed_time < self.vectorized_threshold
            else:
                vectorized_flag = self.vectorized

            def get_logl_from_K(psi_table, K_matrix):
                """
                Convert matrix of K values into log-likelihoods.

                Parameters:
                ----------
                psi_table: np.array
                    The digamma function table.
                K_matrix: np.array
                    The matrix of K values.

                Returns:
                ----------
                logl_matrix: np.array
                    The matrix of log-likelihoods.
                psi_tab: np.array
                    The digamma function table.
                """
                K_max = max(1, np.max(K_matrix))
                if K_max > len(psi_table):  # fill digamma function table
                    psi_table = np.concatenate(
                        (
                            psi_table,
                            psi(1)
                            - psi(np.arange(len(psi_table) + 1, K_max + 1)),
                        )
                    )
                logl_matrix = psi_table[
                    np.maximum(1, K_matrix.astype(int)) - 1
                ]
                return logl_matrix, psi_table

            def vectorized_ibs_sampling(
                params, simulated_data0, elapsed_time0, t0, num_reps
            ):
                """
                A function to perform vectorized Inverse Biased Sampling (IBS) for the given parameters.

                Parameters:
                ----------
                params: np.array
                    The parameter vector.
                simulated_data0: np.array
                    The initial simulation data.
                elapsed_time0: float
                    The initial elapsed time.
                t0: float
                    The starting time of the IBS sampling.
                num_reps: int
                    The number of repetitions for each trial.

                Returns:
                ----------
                neg_logl: float
                    The negative log-likelihood.
                K: np.array
                    A matrix of samples-to-hit for each trial and repeat.
                num_reps_per_trial: np.array
                    The number of repetitions for each trial.
                num_samples_total: int
                    The total number of samples drawn.
                fun_count: int
                    The number of time the sample_from_model function was called.
                exit_flag: int
                    The exit flag (0 = correct termination, 1 = negative log-likelihood threshold reached, 2 = maximum runtime reached, 3 = maximum iterations reached).
                """

                num_trials = self.response_matrix.shape[0]
                trials = np.arange(num_trials)  # enumerate the trials
                num_samples_total = 0  # total number of samples drawn
                fun_count = 0
                psi_table = []
                exit_flag = 0

                # Empty matrix of K values (samples-to-hit) for each repeat for each trial
                K_matrix = np.zeros((num_reps, num_trials), dtype=int)

                # Matrix of rep counts
                K_place0 = np.tile(
                    np.arange(num_reps)[:, np.newaxis], (1, num_trials)
                )

                # Current repetition being sampled for each trial
                repetition = np.zeros(num_trials)

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
                    T = trials[repetition < target_hits]

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
                    num_samples = min(
                        max(1, np.round(samples_level)), self.max_samples
                    )
                    max_samples = np.ceil(self.max_mem / num_considered_trials)
                    num_samples = min(num_samples, max_samples)
                    T_matrix = np.tile(T, (int(num_samples), 1))

                    # Simulate trials
                    if (
                        iter == 0
                        and num_samples == 1
                        and simulated_data0 is not None
                    ):
                        simulated_data = simulated_data0
                        elapsed_time = elapsed_time0
                    else:
                        start = time.time()
                        if self.design_matrix is None:
                            simulated_data = self.sample_from_model(
                                params, T_matrix.reshape(-1)
                            )
                        else:
                            simulated_data = self.sample_from_model(
                                params,
                                self.design_matrix[T_matrix.reshape(-1)],
                            )
                        fun_count += 1
                        elapsed_time = time.time() - start

                    # Check that the returned simulated data have the right size
                    if len(simulated_data) != np.size(T_matrix):
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
                    hits_temp = (
                        self.response_matrix[T_matrix.reshape(-1)]
                        == simulated_data
                    )

                    def get_K_from_hits(hits_temp):

                        # Build matrix of new hits (sandwich with buffer of hits, then removed)
                        hits_new = np.concatenate(
                            (
                                np.ones((1, num_considered_trials)),
                                hits_temp.reshape(T_matrix.shape),
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
                                (
                                    K_iter,
                                    np.zeros(num_considered_trials).reshape(
                                        -1, 1
                                    ),
                                )
                            ),
                            axis=1,
                        )
                        - 1
                    )
                    row_index = np.arange(len(T))
                    # Subtract one hit from last K (it was added)
                    K_iter[row_index, index_last] = (
                        K_iter[row_index, index_last] - 1
                    )
                    K_open[T] = K_iter[row_index, index_last]

                    # For each trial, ignore entries of K_iter past max number of reps
                    index_mat = (
                        np.tile(
                            np.arange(K_iter.shape[1])[:, np.newaxis],
                            (1, num_considered_trials),
                        )
                        + repetition[T]
                    )
                    K_iter[index_mat.T >= num_reps] = 0

                    # Find last K position for each trial again
                    index_last2 = (
                        np.argmin(
                            np.hstack(
                                (
                                    K_iter,
                                    np.zeros(num_considered_trials).reshape(
                                        -1, 1
                                    ),
                                )
                            ),
                            axis=1,
                        )
                        - 1
                    )

                    # Add current K to full K matrix
                    K_iter_place = (
                        K_place0[:, :num_considered_trials] >= repetition[T]
                    ) & (
                        K_place0[:, :num_considered_trials]
                        <= repetition[T] + index_last2
                    )
                    K_place = np.zeros_like(K_place0, dtype=bool)
                    K_place[:, T] = K_iter_place
                    K_mat_flat = K_matrix.flatten("F")
                    K_mat_flat[K_place.flatten("F")] = K_iter[
                        K_iter > 0
                    ].flatten()
                    K_matrix = K_mat_flat.reshape(K_matrix.shape, order="F")
                    # update current repetitions
                    repetition[T] = repetition[T] + index_last

                    # Compute log-likelihood only if a threshold is set
                    if np.isfinite(self.neg_logl_threshold):
                        R_min = np.min(repetition[T])
                        if R_min >= K_matrix.shape[0]:
                            continue
                        logl_temp, psi_table = get_logl_from_K(
                            psi_table, K_matrix[int(R_min), :]
                        )
                        nLL_temp = -np.sum(logl_temp, axis=0)
                        if nLL_temp > self.neg_logl_threshold:
                            index_move = repetition == R_min
                            repetition[index_move] = R_min + 1
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

                if np.isfinite(self.neg_logl_threshold) and exit_flag == 1:
                    try:
                        raise RuntimeWarning(
                            "Warning in IBS execution: termination after negative log-likelihood threshold was reached (the estimate is biased)"
                        )
                    except RuntimeWarning as e:
                        print(e)

                # Compute log-likelihood
                num_reps_per_trial = np.sum(
                    K_matrix > 0, axis=0
                )  # number of repetitions of the single trials
                logl_matrix, psi_table = get_logl_from_K(psi_table, K_matrix)
                neg_logl = -np.sum(logl_matrix, axis=0) / num_reps_per_trial
                K = K_matrix.T

                return (
                    neg_logl,
                    K,
                    num_reps_per_trial,
                    num_samples_total,
                    fun_count,
                    exit_flag,
                )

            def loop_ibs_sampling(params, simulated_data0, t0, num_reps):
                """
                A function to perform loop-based Inverse Biased Sampling (IBS) for the given parameters.

                Parameters:
                ----------
                params: np.array
                    The parameter vector.
                simulated_data0: np.array
                    The initial simulation data.
                t0: float
                    The starting time of the IBS sampling.
                num_reps: int
                    The number of repetitions for each trial.

                Returns:
                ----------
                neg_logl: float
                    The negative log-likelihood.
                K: np.array
                    A matrix of samples-to-hit for each trial and repeat.
                num_reps_per_trial: np.array
                    The number of repetitions for each trial.
                num_samples_total: int
                    The total number of samples drawn.
                fun_count: int
                    The number of time the sample_from_model function was called.
                exit_flag: int
                    The exit flag (0 = correct termination, 1 = negative log-likelihood threshold reached, 2 = maximum runtime reached, 3 = maximum iterations reached).
                """

                num_trials = self.response_matrix.shape[0]

                trials = np.arange(num_trials)  # enumerate the trials
                max_iter = self.max_iter

                K = np.zeros(
                    (num_trials, num_reps)
                )  # saves the number of iterations needed for the sample to match the trial response
                num_samples_total = 0  # total number of samples drawn
                fun_count = 0
                psi_table = []
                exit_flag = 0

                for i_Rep in range(num_reps):

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
                        if (
                            iter == 0
                            and i_Rep == 0
                            and simulated_data0 is not None
                        ):
                            simulated_data = simulated_data0
                            fun_count += 1
                        elif self.design_matrix is None:
                            # call function with input params only for the trials that have not been hit yet
                            simulated_data = self.sample_from_model(params, T)
                        else:
                            # call function with input params and design_mat only for the trials that have not been hit yet
                            simulated_data = self.sample_from_model(
                                params, self.design_matrix[T]
                            )
                            fun_count += 1

                        if np.shape(np.atleast_1d(simulated_data))[0] != len(
                            T
                        ):
                            raise ValueError(
                                "IBS: number of rows of returned simulated data does not match the number of requested trials"
                            )
                        num_samples_total += len(T)
                        hits_new = simulated_data == self.response_matrix[T]
                        hits[T] = hits_new

                        K[np.atleast_1d(T)[hits_new], i_Rep] = offset

                        if np.isfinite(self.neg_logl_threshold):
                            K[hits == False, i_Rep] = offset
                            logl_matrix, psi_table = get_logl_from_K(
                                psi_table, K[:, i_Rep]
                            )
                            neg_logl = -np.sum(
                                logl_matrix, axis=0
                            )  # compute the negative log-likelihood of the current repetition
                            if neg_logl > self.neg_logl_threshold:
                                T = []
                                exit_flag = 1
                                break
                        offset += 1

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
                logl_matrix, psi_table = get_logl_from_K(psi_table, K)
                neg_logl = -np.sum(logl_matrix, axis=1) / num_reps_per_trail

                return (
                    neg_logl,
                    K,
                    num_reps_per_trail,
                    num_samples_total,
                    fun_count,
                    exit_flag,
                )

            if vectorized_flag:
                (
                    neg_logl,
                    K,
                    num_reps_per_trial,
                    num_samples_total,
                    fun_count,
                    exit_flag,
                ) = vectorized_ibs_sampling(
                    params, simulated_data, elapsed_time, t0, num_reps
                )
            else:
                (
                    neg_logl,
                    K,
                    num_reps_per_trial,
                    num_samples_total,
                    fun_count,
                    exit_flag,
                ) = loop_ibs_sampling(params, simulated_data, t0, num_reps)

            neg_logl = np.sum(neg_logl * weights)
            if return_positive:
                neg_logl = -neg_logl
            return (
                neg_logl,
                K,
                num_reps_per_trial,
                num_samples_total,
                fun_count,
                exit_flag,
            )

        (
            neg_logl,
            K,
            num_reps_per_trial,
            num_samples_total,
            fun_count,
            exit_flag,
        ) = compute_logl(self, params, num_reps, weights, return_positive, t0)

        if additional_output in [None, "none"]:
            return neg_logl
        elif additional_output not in [None, "none"]:
            # compute variance of log-likelihood
            K_max = np.amax(K, initial=1)
            K_tab = -polygamma(1, np.arange(1, K_max + 1)) + polygamma(1, 1)
            logl_var = K_tab[np.maximum(1, K.astype(int)) - 1]
            neg_logl_var = np.sum(logl_var, axis=1) / num_reps_per_trial**2
            neg_logl_var = np.sum(neg_logl_var * (weights**2))
            if additional_output == "var":
                return neg_logl, neg_logl_var
            if additional_output == "std":
                return neg_logl, np.sqrt(neg_logl_var)
            if additional_output == "full":
                message = ""
                if exit_flag == 0:
                    message = "Correct termination (the estimate is unbiased)."
                elif exit_flag == 1:
                    message = "Termination after negative log-likelihood threshold was reached (the estimate is biased)."
                elif exit_flag == 2:
                    message = "Termination after maximum execution time was reached (the estimate can be arbitrarily biased)."
                elif exit_flag == 3:
                    message = "Termination after maximum number of iterations was reached (the estimate can be arbitrarily biased)."

                return EstimateResult(
                    neg_logl=neg_logl,
                    neg_logl_var=neg_logl_var,
                    neg_logl_std=np.sqrt(neg_logl_var),
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
