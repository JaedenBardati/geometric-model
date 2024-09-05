# A small python package that handles geometric models and their sampling and fitting. 
# This can be used to fit simulation data to SKIRT radiative transfer geometric models 
# to speed up the RT simulations, obtain insights on the data, and compare best fit 
# geometric models. 
# Jaeden Bardati
# Requires: numpy, scipy, matplotlib

import abc
import warnings

import numpy as np
import scipy
import scipy.spatial
import scipy.optimize
import matplotlib
import matplotlib.pyplot as plt


######################################
########## Abstract Classes ##########
######################################

class GeometricModel(metaclass=abc.ABCMeta):
    """
    An abstract class for building geometric models to fit or sample.
    """
    
    @property
    @abc.abstractmethod
    def VARIABLE_NAMES(self):
        pass

    @property
    @abc.abstractmethod
    def ARGUMENT_NAMES(self):
        pass

    @property
    @abc.abstractmethod
    def DEFAULT_FIT_BOUNDS(self):
        pass

    @property
    @abc.abstractmethod
    def DEFAULT_FIT_INITIAL_VALUES(self):
        pass

    @abc.abstractmethod
    def _model_function(self, *model_args):
        pass

    @staticmethod
    @abc.abstractmethod
    def _default_argument_sampling_function(*sample_args, **sample_kwargs):
        pass

    
    def __init__(self, *model_variables):
        self._subclass_name_ = type(self).__name__
        self._variables = tuple(model_variables)
        assert len(self.VARIABLE_NAMES) == len(model_variables), "Variables constant does not match the model arguments in the <%s> class definition." % self._subclass_name_
        
        self._fixed_variables = set([var for var, val in zip(self.VARIABLE_NAMES, self._variables) if val is not None])
        self._free_variables = set([var for var, val in zip(self.VARIABLE_NAMES, self._variables) if val is None])
        self._fitted_variables = set()
        self._fit_covariance_matrix = None
        self._extra_fit_info = None
        assert len(self._fixed_variables.intersection(self._free_variables)) == 0, "There is an overlap between the fixed and free variables."
    
    def __call__(self, *model_args, **free_variables):
        free_variables_difference = self._free_variables.symmetric_difference(free_variables.keys())
        if len(free_variables_difference) != 0:
            for f_var in free_variables_difference:
                if f_var in self._free_variables:
                    raise RuntimeError("Missing input for free variable '{}'.".format(f_var))
                else:
                    raise TypeError("{}.__call__() got an unexpected keyword argument '{}'. There is no such free variable defined.".format(self._subclass_name_, f_var))
        self._variables = [val if var in self._fixed_variables else free_variables[var] for var, val in zip(self.VARIABLE_NAMES, self._variables)]
        return self._model_function(*model_args)

    
    def fit(self, fit_data, *model_args, bounds='default', p0='default', sigma=None, x_scale=None, full_output=False, **other_curvefit_kwargs):
        # Pass the data to fit and the model args. 
        # Pass x_scale='default' to use the class-defined scale, x_scale=None for no scale, and x_ 
        if len(self._free_variables) < 1:
            raise Exception("No free variables to fit!")

        # create some model inputs and useful variables
        variable_dict = {var: val for var, val in zip(self.VARIABLE_NAMES, self._variables)}
        ordered_free_variables = [var for var in self.VARIABLE_NAMES if var in self._free_variables]

        if bounds == 'default':
            lower_bounds = [lbound for lbound, var in zip(self.DEFAULT_FIT_BOUNDS[0], self.VARIABLE_NAMES) if var in self._free_variables]
            upper_bounds = [ubound for ubound, var in zip(self.DEFAULT_FIT_BOUNDS[1], self.VARIABLE_NAMES) if var in self._free_variables]
            bounds = (lower_bounds, upper_bounds)
        
        if p0 == 'default':
            p0 = [scale_element for scale_element, var in zip(self.DEFAULT_FIT_INITIAL_VALUES, self.VARIABLE_NAMES) if var in self._free_variables]
        
        # support for bounds that are defined by other variables by defining new variables
        if bounds is not None:
            if len(bounds) != 2:
                raise ValueError("Bounds must be in a 2-tuple containing the lists of lower and upper bounds to the free variables, respectively, in order.")
            if len(bounds[0]) != len(self._free_variables) or len(bounds[1]) != len(self._free_variables):
                raise ValueError("Upper and lower bounds must have the same length as the free variables.")
            
            _, unique_variable_counts = np.unique([v for v in np.ravel(bounds) if v in ordered_free_variables], return_counts=True)
            if unique_variable_counts.size != 0 and np.any(unique_variable_counts > 1): # if any doubles of variable type elements
                raise ValueError("Unrecognizable variable bounds structure. Duplicates of variables in the bounds is not supported.")
                # Note: this is very likely too restrictive! If you encounter an error, try deleting this and seeing if it still works.
    
            root_variable_indexes = [None for _ in ordered_free_variables]
            # ^ array index: old free variable (x), items: associated root variable (r), s.t. new variable is v = x - r > 0
            redefined_variable_indexes = []
            for i, (lbound, ubound) in enumerate(zip(*bounds)):
                # replace fixed variables in bounds with stored numbers
                if lbound in self._fixed_variables:
                    bounds[0][i] = variable_dict[lbound]
                if ubound in self._fixed_variables:
                    bounds[1][i] = variable_dict[ubound]
                
                # replace free variables in bounds by redefining the variables
                is_var_lbound, is_var_ubound = lbound in ordered_free_variables, ubound in ordered_free_variables
                if is_var_lbound and is_var_ubound:
                    raise TypeError("Double dependent bounds are currently not supported.")
                
                if is_var_ubound:
                    j = ordered_free_variables.index(ubound)
                    if ordered_free_variables[i] != bounds[0][j]:
                        raise ValueError("Variable '{}' must referenced as a minimum in the bounds of variable '{}', since the contrapositive is true.".format(ordered_free_variables[i], ubound))
                    if j not in redefined_variable_indexes:
                        root_variable_indexes[j] = i
                        redefined_variable_indexes.append(j)
                
                if is_var_lbound:
                    j = ordered_free_variables.index(lbound)
                    if ordered_free_variables[i] != bounds[1][j]:
                        raise ValueError("Variable '{}' must referenced as a maximum in the bounds of variable '{}', since the contrapositive is true.".format(ordered_free_variables[i], lbound))
                    if ubound != np.inf and ubound is not None: 
                        raise ValueError("Since variable '{}' has lower bound variable {}, it must be bounded at infinity. Support is not currently provided for finitely upper bounded variable dependent bounds.".format(ordered_free_variables[i], lbound))
                        # Note: If you want to implement this, you have to get around the r+v<MAX, r>0, v>0 triangle constraint, possibly with some non-linear transformation. 
                    if j not in root_variable_indexes:
                        root_variable_indexes[i] = j
                        redefined_variable_indexes.append(i)

            # actually change the bounds due to the change in the variables
            for old_var_index, root_index in enumerate(root_variable_indexes):
                if root_index is not None:
                    bounds[1][root_index] = bounds[1][old_var_index] # i.e. np.inf
                    bounds[0][old_var_index] = 0.0
                    # Note: bounds[1][old_var_index] says np.inf still, despite the change of variable, due to the nature of infinity (see other note above).
                    if p0 is not None:
                        if p0[root_index] < p0[old_var_index]:
                            p0[old_var_index] = p0[old_var_index] - p0[root_index]
                        if p0[root_index] == p0[old_var_index]:
                            p0[old_var_index] = p0[old_var_index]/10. # 1 ofm smaller than given (since it is now the difference)
                        else:
                            raise ValueError("The initial value of free variable '{}' (index {}) is larger than the initial value of free variable '{}' (index {}), despite the bounds requiring '{}' to be smaller than '{}'.".format(ordered_free_variables[root_index], ordered_free_variables[old_var_index], root_index, old_var_index, ordered_free_variables[root_index], ordered_free_variables[old_var_index]))
                    if sigma is not None:
                        if np.asarray(sigma).shape == 0:
                            pass
                        elif sigma.shape == (len(ordered_free_variables),):
                            sigma[old_var_index] = np.sqrt(sigma[old_var_index]**2 + sigma[root_index]**2)
                        elif len(sigma.shape) == (len(ordered_free_variables), len(ordered_free_variables)):
                            extra_sigma = np.zeros(sigma.shape)
                            extra_sigma[old_var_index, old_var_index] += sigma[root_index, root_index]
                            for t in range(sigma.shape[0]):
                                extra_sigma[old_var_index, t] += sigma[root_index, t]
                                extra_sigma[t, old_var_index] += sigma[t, root_index]
                            sigma = sigma + extra_sigma
                        else:
                            raise ValueError("Must enter either a scalar, an N sized array or an NxN matrix for sigma, where N is the number of free variables.")
                    if x_scale is not None and x_scale != 'jac':
                        if x_scale[root_index] < x_scale[old_var_index]:
                            x_scale[old_var_index] = x_scale[old_var_index] - x_scale[root_index]
                        if x_scale[root_index] == x_scale[old_var_index]:
                            x_scale[old_var_index] = x_scale[old_var_index]/10.  # 1 ofm smaller than given (since it is now the difference)
                        else:
                            raise ValueError("The scale of free variable '{}' is larger than the scale of free variable '{}' (index {}), despite the bounds requiring '{}' (index {}) to be smaller than '{}'.".format(ordered_free_variables[root_index], ordered_free_variables[old_var_index], root_index, old_var_index, ordered_free_variables[root_index], ordered_free_variables[old_var_index]))
            
        # create model function call for scipy
        def m_foo(model_args, *free_variables):
            # convert from new artificial variables from bound change back to the model's free variables (x = r + v)
            free_variables = {f_var: f_val if root_var_index is None else free_variables[root_var_index] + f_val for f_var, f_val, root_var_index in zip(ordered_free_variables, free_variables, root_variable_indexes)}
            return self(*model_args, **free_variables)
        
        # hack to allow for a variable amount of free variables
        pre_fullargspec = scipy.optimize._minpack_py._getfullargspec
        args = ['self'] + ordered_free_variables  # list(self.ARGUMENT_NAMES)
        def my_fullargspec(func):
            pre_fas = pre_fullargspec(func)
            return scipy._lib._util.FullArgSpec(args, [], pre_fas.varkw, pre_fas.defaults, pre_fas.kwonlyargs,
                               pre_fas.kwonlydefaults, pre_fas.annotations)
        
        # run model fitting
        scipy.optimize._minpack_py._getfullargspec = my_fullargspec
        if x_scale is None:
            if bounds is None:
                r = scipy.optimize.curve_fit(m_foo, model_args, fit_data, p0=p0, sigma=sigma, full_output=full_output, **other_curvefit_kwargs)
            else:
                r = scipy.optimize.curve_fit(m_foo, model_args, fit_data, p0=p0, sigma=sigma, bounds=bounds, full_output=full_output, **other_curvefit_kwargs)
        else:
            if bounds is None:
                r = scipy.optimize.curve_fit(m_foo, model_args, fit_data, p0=p0, sigma=sigma, x_scale=x_scale, full_output=full_output, **other_curvefit_kwargs)
            else:
                r = scipy.optimize.curve_fit(m_foo, model_args, fit_data, p0=p0, sigma=sigma, bounds=bounds, full_output=full_output, x_scale=x_scale, **other_curvefit_kwargs)
        scipy.optimize._minpack_py._getfullargspec = pre_fullargspec
        if full_output:
            popt, pcov, infodict, mesg, ier = r
            self._extra_fit_info = (infodict, mesg, ier)
        else:
            popt, pcov = r
        
        # convert from new artificial variables from bound change back to the model's free variables (x = r + v)
        for old_var_index, root_index in enumerate(root_variable_indexes):
            if root_index is not None:
                popt[old_var_index] = popt[root_index] + popt[old_var_index]
                
                extra_pcov = np.zeros(pcov.shape)
                extra_pcov[old_var_index, old_var_index] += pcov[root_index, root_index]
                for t in range(pcov.shape[0]):
                    extra_pcov[old_var_index, t] += pcov[root_index, t]
                    extra_pcov[t, old_var_index] += pcov[t, root_index]
                pcov = pcov + extra_pcov

        self._variables = tuple([var if var not in ordered_free_variables else popt[ordered_free_variables.index(var)] for var in self._variables])
        self._free_variables = set()
        self._fixed_variables = set(self.VARIABLE_NAMES)
        self._fitted_variables = set(ordered_free_variables)
        self._fit_covariance_matrix = pcov

        return r

    
    def sample(self, *sample_args, argument_sampling_function=None, **sample_kwargs):
        if len(self._free_variables) != 0:
            raise RuntimeError("You cannot sample from a model that has free variables. Define the free variables first by defining the variables or fitting them from data.")
        if argument_sampling_function is None:
            argument_sampling_function = self._default_argument_sampling_function
        _model_args = argument_sampling_function(*sample_args, **sample_kwargs)
        return (*_model_args, self._model_function(*_model_args))

    
    @property
    def variables(self):
        return {var: val for var, val in zip(self.VARIABLE_NAMES, self._variables)}
    
    @property
    def fixed_variables(self):
        return {var: val for var, val in zip(self.VARIABLE_NAMES, self._variables) if var in self._fixed_variables}

    @property
    def free_variables(self):
        return {var: val for var, val in zip(self.VARIABLE_NAMES, self._variables) if var in self._free_variables}
    
    @property
    def fitted_variables(self):
        return {var: val for var, val in zip(self.VARIABLE_NAMES, self._variables) if var in self._fitted_variables}

    @property
    def ordered_fixed_variables(self):
        return [var for var in self.VARIABLE_NAMES if var in self._fixed_variables]

    @property
    def ordered_free_variables(self):
        return [var for var in self.VARIABLE_NAMES if var in self._free_variables]
    
    @property
    def ordered_fitted_variables(self):
        return [var for var in self.VARIABLE_NAMES if var in self._fitted_variables]
    
    @property
    def fitted_variable_errors(self):
        return {variable: np.sqrt(variance) for variable, variance in zip(self.ordered_fitted_variables, np.diag(self._fit_covariance_matrix))}

    @property
    def fit_covariance_matrix(self):
        return self._fit_covariance_matrix

    @property
    def extra_fit_info(self):
        return self._extra_fit_info

    @property
    def fit_cond(self):
        # if this is very large, the model is likely overparametrized. 
        return np.linalg.cond(self._fit_covariance_matrix)


########################################
########## Sampling Functions ##########
########################################

## BETTER WAY TO DO THIS: Make a sampler defined by a coordinate system + distribution function, coordinate system class is:
# def CoordinateSystem(metaclass=abc.ABCMeta):
#     """
#     A abstract class to handle coordinate system transformation and sampling.
#     """
#     pass


def _uniform_cartesian(N, dim, lo_bound=None, hi_bound=None, center=None, size=None):
    # Returns an evenly distributed cartesian coordinate system of "particles".
    if hi_bound is None and lo_bound is None and center is None and size is not None:
        center = 0.0  # default to zero being the center if only size is entered
    if ((lo_bound is not None and hi_bound is not None and center is None and size is None) == 
        (lo_bound is None and hi_bound is None and center is not None and size is not None)):
        raise ValueError("Normalization must be entered either from both two bounds or from both the center and size.")
    if center is None:
        lo_bound, hi_bound = np.asarray(lo_bound), np.asarray(hi_bound)
        if not np.all(hi_bound > lo_bound):
            raise ValueError("The upper bounds should be larger than the lower bounds.")
        size = hi_bound - lo_bound
        center = lo_bound + size/2.0
    else:
        size, center = np.asarray(size), np.asarray(center)
    if len(size.shape) > 1 or len(center.shape) > 1:
        raise ValueError("Unrecognized format for one or more bound inputs.")
    if not len(size.shape):
        size = np.repeat(size, dim)
    if not len(center.shape):
        center = np.repeat(center, dim)
    size = np.repeat(size, N).reshape((dim, N))
    center = np.repeat(center, N).reshape((dim, N))
    return size*(np.random.rand(dim, N) - 0.5) + center

def _uniform_spherical(N, lo_bound=None, hi_bound=None, center=None, size=None):
    # Returns an evenly distributed spherical coordinate system of "particles".
    x, y, z = _uniform_cartesian(N, 3, lo_bound=lo_bound, hi_bound=hi_bound, center=center, size=size)
    r, theta, phi = cart_to_sph(x, y, z)
    return r, theta, phi

def _uniform_cylindrical(N, lo_bound=None, hi_bound=None, center=None, size=None):
    # Returns an evenly distributed spherical coordinate system of "particles".
    x, y, z = _uniform_cartesian(N, 3, lo_bound=lo_bound, hi_bound=hi_bound, center=center, size=size)
    s, phi, z = cart_to_cyl(x, y, z)
    return s, phi, z


uniform_x = lambda N, **kwargs: tuple(_uniform_cartesian(N, 1, **kwargs))
uniform_x_y = lambda N, **kwargs: tuple(_uniform_cartesian(N, 2, **kwargs))
uniform_x_y_z = lambda N, **kwargs: tuple(_uniform_cartesian(N, 3, **kwargs))
uniform_x_y_z_t = lambda N, **kwargs: tuple(_uniform_cartesian(N, 4, **kwargs))

uniform_r = uniform_x     # no different from sampling uniform x
uniform_r_phi = lambda N, **kwargs: tuple(_uniform_spherical(N, **kwargs)[i] for i in [0, 2])
uniform_r_theta = lambda N, **kwargs: tuple(_uniform_spherical(N, **kwargs)[i] for i in [0, 1])
uniform_r_theta_phi = lambda N, **kwargs: tuple(_uniform_spherical(N, **kwargs))

uniform_s = uniform_x     # no different from sampling uniform x
uniform_s_phi = lambda N, **kwargs: tuple(_uniform_cylindrical(N, **kwargs)[i] for i in [0, 1])
uniform_s_z = lambda N, **kwargs: tuple(_uniform_cylindrical(N, **kwargs)[i] for i in [0, 2])
uniform_s_phi_z = lambda N, **kwargs: tuple(_uniform_cylindrical(N, **kwargs))


#######################################
########## Geometric Classes ##########
#######################################

class PowerLawSphere(GeometricModel):
    VARIABLE_NAMES = ('norm', 'index', 'rmax')
    ARGUMENT_NAMES = ('r', 'theta', 'phi')
    DEFAULT_FIT_BOUNDS = ([0.0, -np.inf, 0.0], 
                          [np.inf, np.inf, np.inf])
    DEFAULT_FIT_INITIAL_VALUES = (1., 0., 1.)
    
    def __init__(self, norm, index, rmax):
        super().__init__(norm, index, rmax)
        
    def _model_function(self, r, theta=None, phi=None):
        _norm, _index, _rmax = self._variables
        _f = _norm * np.power(r, -_index)
        _l = (r < _rmax)
        return _f*_l

    @staticmethod
    def _default_argument_sampling_function(*sample_args, **sample_kwargs):
        return uniform_r_theta_phi(*sample_args, **sample_kwargs)


class PowerLawTorus(GeometricModel):
    # Make to match https://skirt.ugent.be/skirt9/class_torus_geometry.html
    # Opening angle is in degrees, but theta is in radians (polar angle).
    
    VARIABLE_NAMES = ('norm', 'radial_index', 'polar_index', 'rmin', 'rmax', 'opening_angle')
    ARGUMENT_NAMES = ('r', 'theta', 'phi')
    DEFAULT_FIT_BOUNDS = ([0.0, -np.inf, -np.inf, 0.0, 'rmin', 0.0], 
                          [np.inf, np.inf, np.inf, 'rmax', np.inf, 180.0])
    DEFAULT_FIT_INITIAL_VALUES = (1., 0., 0., 1., 2., 10.)
    
    def __init__(self, norm, radial_index, polar_index, rmin, rmax, opening_angle):
        super().__init__(norm, radial_index, polar_index, rmin, rmax, opening_angle)
        
    def _model_function(self, r, theta, phi=None):
        _norm, _radial_index, _polar_index, _rmin, _rmax, _opening_angle = self._variables
        _opening_angle_rad = _opening_angle*np.pi/180.
        _f = _norm * np.power(r, -_radial_index) * np.exp(-_polar_index * np.abs(np.cos(theta)))
        _l = (_rmin < r) * (r < _rmax) * (np.pi/2. - _opening_angle_rad < theta) * (theta < np.pi/2. + _opening_angle_rad)
        return _f*_l

    @staticmethod
    def _default_argument_sampling_function(*sample_args, **sample_kwargs):
        return uniform_r_theta_phi(*sample_args, **sample_kwargs)


class PowerLawTorusSmooth(GeometricModel):
    # Make to be similar to PowerLawTorus, but with an smoothing parameter determined by the cutoff indexes. This helps with model fitting.
    # Opening angle is in degrees, but theta is in radians (polar angle).
    # To approach a sharp power law torus, the smoothing index (k) should be large.
    # For k>~100, you get a sharp cutoff. For k ~ 10, you get a "fuzzy" torus. For 0 < k <~ 3, it ceases to become a clear torus.
    # The normalization scheme determines how the function behaves at small smoothing indexes (k<~10). Implemented options are:
    #   - 'fixed_max': Enforces the maximum of the smoothing function to be 1, such that the maximum possible model output is always the norm entered.
    #   - 'inf_norm': Enforces the integral of the smoothing function from -inf and inf to be constant, such that as the smoothing index approaches inf, the maximum possible model output is the norm entered.
    
    VARIABLE_NAMES = ('norm', 'radial_index', 'polar_index', 'rmin', 'rmax', 'opening_angle', 'radial_cutoff_index', 'polar_cutoff_index')
    ARGUMENT_NAMES = ('r', 'theta', 'phi')
    DEFAULT_FIT_BOUNDS = ([0.0, -np.inf, -np.inf, 0.0, 'rmin', 0.0, 3.0, 3.0], 
                          [np.inf, np.inf, np.inf, 'rmax', np.inf, 90.0, np.inf, np.inf])
    DEFAULT_FIT_INITIAL_VALUES = (1., 0., 0., 1., 2., 10., 250., 250.)
    
    NORM_SCHEMES = ('fixed_max', 'inf_norm')
    NORM_SCHEME_DEFAULT = 'fixed_max'
    
    def __init__(self, norm, radial_index, polar_index, rmin, rmax, opening_angle, radial_cutoff_index, polar_cutoff_index, norm_scheme=NORM_SCHEME_DEFAULT):
        super().__init__(norm, radial_index, polar_index, rmin, rmax, opening_angle, radial_cutoff_index, polar_cutoff_index)
        self._norm_scheme = norm_scheme

    @staticmethod
    def smooth_cutoff_function_normalization(xlo, xhi, smooth_index, norm_scheme=NORM_SCHEME_DEFAULT):
        if norm_scheme == 'fixed_max':
            return (1+np.exp(-0.5*smooth_index))**2
        elif norm_scheme == 'inf_norm':
            _a = smooth_index*(np.exp(-smooth_index)-1.0)
            _b = 2.0*(smooth_index*xlo/(xhi - xlo) + np.log(np.exp(-0.5*smooth_index)+1.0) - np.log(np.exp(0.5*smooth_index*(xlo+xhi)/(xhi - xlo)) + np.exp(smooth_index*xlo/(xhi - xlo))))
            return _a/_b
        else:
            raise ValueError("The normalization scheme entered is not recognized.")
    
    @staticmethod
    def smooth_cutoff_function(x, xlo, xhi, smooth_index, norm_scheme=NORM_SCHEME_DEFAULT):
        _a = PowerLawTorusSmooth.smooth_cutoff_function_normalization(xlo, xhi, smooth_index, norm_scheme=norm_scheme)
        _b = (1.0+np.exp(smooth_index*(xlo-x)/(xhi - xlo)))*(1.0+np.exp(smooth_index*(x-xhi)/(xhi - xlo)))
        return _a/_b
    
    def _model_function(self, r, theta, phi=None):
        _norm, _radial_index, _polar_index, _rmin, _rmax, _opening_angle, _radial_cutoff_index, _polar_cutoff_index = self._variables
        _opening_angle_rad = _opening_angle*np.pi/180.
        _f = _norm * np.power(r, -_radial_index) * np.exp(-_polar_index * np.abs(np.cos(theta)))
        _r = PowerLawTorusSmooth.smooth_cutoff_function(r, _rmin, _rmax, _radial_cutoff_index, norm_scheme=self._norm_scheme)
        _t = PowerLawTorusSmooth.smooth_cutoff_function(theta, np.pi/2. - _opening_angle_rad, np.pi/2. + _opening_angle_rad, _polar_cutoff_index, norm_scheme=self._norm_scheme)
        return _f*_r*_t

    @staticmethod
    def _default_argument_sampling_function(*sample_args, **sample_kwargs):
        return uniform_r_theta_phi(*sample_args, **sample_kwargs)



################################################
########## Coordinate Transformations ##########
################################################

def cart_to_sph(x, y, z):
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi

def cart_to_cyl(x, y, z):
    s = np.sqrt(x*x + y*y)
    phi = np.arctan2(y, x)
    return s, phi, z

def sph_to_cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def cyl_to_cart(r, phi, z):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, z


#####################################
########## Noise Functions ##########
#####################################

def poisson_noise(size, noise_amplitude=1.0, lam=1.0):
    return noise_amplitude * np.random.poisson(lam=lam, size=size)

def gaussian_noise(size, noise_amplitude=1.0):
    return noise_amplitude * np.random.normal(size=size)

def shot_noise(size, snr=10.0, signal_amplitude=1.0):
    return poisson_noise(size, noise_amplitude=signal_amplitude/snr**2, lam=snr)

    
########################################
########## Plotting Functions ##########
########################################

def quick_slice_plot(x, y, z, data, eps=0.01, lim=10, ret_fig=False, log_scale=False, cmap='viridis', coords='cart', min=None, max=None, s=4):
    if coords == 'sph':
        x, y, z = sph_to_cart(x, y, z)
    elif coords == 'cyl':
        x, y, z = cyl_to_cart(x, y, z)
    elif coords != 'cart':
        warnings.warn('Unknown coordinate system provided. Defaulting to standard axis cuts as though it is Cartesian.')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    if log_scale:
        data_gtr_0 = data>0.0
        if not np.all(data_gtr_0):
            data = data[data_gtr_0]
            x = x[data_gtr_0]
            y = y[data_gtr_0]
            z = z[data_gtr_0]
            warnings.warn("Some of the data has been cut, since it is less than 0 and a log scale colorbar is set.")
    slice_0 = np.logical_and(-eps<z, z<eps)
    slice_1 = np.logical_and(-eps<x, x<eps)
    slice_2 = np.logical_and(-eps<y, y<eps)
    norm_min = min if min is not None else np.min([data[slice_0].min(), data[slice_1].min(), data[slice_2].min()])
    norm_max = max if max is not None else np.max([data[slice_0].max(), data[slice_1].max(), data[slice_2].max()])
    matplotlib_normalizer = matplotlib.colors.LogNorm if log_scale else matplotlib.colors.Normalize
    norm = matplotlib_normalizer(norm_min, norm_max)
    im = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    axes[0].scatter(x[slice_0], y[slice_0], c=data[slice_0], s=s, cmap=cmap, norm=norm)
    axes[1].scatter(y[slice_1], z[slice_1], c=data[slice_1], s=s, cmap=cmap, norm=norm)
    axes[2].scatter(x[slice_2], z[slice_2], c=data[slice_2], s=s, cmap=cmap, norm=norm)
    axes[0].set_xlim(-lim, lim)
    axes[0].set_ylim(-lim, lim)
    axes[1].set_xlim(-lim, lim)
    axes[1].set_ylim(-lim, lim)
    axes[2].set_xlim(-lim, lim)
    axes[2].set_ylim(-lim, lim)
    fig.colorbar(im, ax=axes.ravel().tolist())
    if ret_fig:
        return fig


