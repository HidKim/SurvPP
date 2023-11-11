# Python Code for Survival Permanental Process 
This library provides survival permanental process (SurvPP) implemented in Tensorflow. SurvPP provides a scalable Bayesian framework for survival analysis with time-varying covariates. For details, see our NeurIPS2023 paper [1].

The code was tested on Python 3.10.8, tensorflow-deps 2.10.0, tensorflow-macos 2.10.0, and tensorflow-metal 0.6.0.

# Installation
To install latest version:
```
pip install git+https://github.com/HidKim/SurvPP
```

# Basic Usage
Import SurvPP class:
```
from HidKim_SurvPP import survival_permanental_process as SurvPP
```
Initialize SurvPP:
```
model = SurvPP(kernel='Gaussian', eq_kernel='RFM',  
            eq_kernel_options={'n_rfm':500})
```
- `kernel`: *string, default='Gaussian'* <br> 
  >The kernel function for Gaussian process. Only 'Gaussian' is available now.
- `eq_kernel`:  *string, default='RFM'* <br>
  >The approach to constructing equivalent kernel. Only 'RFM' is available now.  
- `eq_kernel_options`:  *dict, default={'n_rfm':500}* <br>
  >The options for constructing equivalent kernel.
  >* `'n_rfm'`: Number of feature samples for the random feature map approach to constructing equivalent kernel.
  
Fit APP with data:
```
time = model.fit(formula, df, set_par, display=True)
```
- `formula`: *ndarray of shape (n_samples, dim_observation)* <br> 
  >The point event data.
- `df`:  *ndarray of shape (dim_observation, 2)*  <br>
  > The hyper-rectangular region of observation. For example, [[x0,x1],[y0,y1]] for two-dimensional region.  
- `cov_fun`: *callable* <br> 
  >The covariate map "cov_fun(t) -> y", where t is a point in the obervation domain, and y is the covariate value at the point.  
- `set_par`:  *ndarray of shape (n_candidates, dim_hyperparameter)*  <br>
  >The set of hyper-parameters for hyper-parameter optimization. The optimization is performed by maximizing the marginal likelihood.
- `display`:  *bool, default=True*  <br>
  >Display the summary of the data and the fitting. 
- **Return**: *float* <br>
  >The execution time.

Predict point process intensity as function of covariates:
```
r_est = model.predict(y, conf_int=[0.025,0.5,0.975])
```
- `y`: *ndarray of shape (n_points, dim_covariate)* <br> 
  >The points on covariate domain for evaluating intensity values.
- `conf_int`:  *ndarray of shape (n_quantiles,), default=[.025, .5, .975]*  <br>
  > The quantiles for predicted intenity.
- **Return**: *ndarray of shape (n_quantiles, n_points)* <br>
  >The predicted values of intensity at the specified points.

# Reference
1. Hideaki Kim. "Survival Permanental Processes for Survival Analysis with Time-Varying Covariates", *Advances in Neural Information Processing Systems 36*, 2023.
```
@inproceedings{kim2023survival,
  title={Survival Permanental Processes for Survival Analysis with Time-Varying Covariates},
  author={Kim, Hideaki},
  booktitle={Advances in Neural Information Processing Systems 36},
  year={2023}
}
``` 

# License
Released under "SOFTWARE LICENSE AGREEMENT FOR EVALUATION". Be sure to read it.

# Contact
Feel free to contact the author Hideaki Kim (hideaki.kin@ntt.com).