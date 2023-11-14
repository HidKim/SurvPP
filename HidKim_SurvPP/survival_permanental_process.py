import numpy as np
from numpy.random import default_rng
import sys, time, os
import tensorflow as tf
from scipy.stats import gamma
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TYP = 'float64'
name = 'survival_permanental_process'

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

class optimizer_adam:
    
    def __init__(self, lr=0.001, beta_1=0.9, 
                 beta_2=0.999, epsilon=1e-07, dtype=tf.float64):
        
        self.lr, self.beta_1, self.beta_2, self.epsilon \
            = tf.cast(lr,dtype), tf.cast(beta_1,dtype), \
            tf.cast(beta_2,dtype), tf.cast(epsilon,dtype)
        self.dtype= dtype

    def reset(self, func, x):
        
        m = tf.zeros(tf.shape(x),self.dtype)
        v = tf.zeros(tf.shape(x),self.dtype)
        x = tf.cast(x,self.dtype)

        return tf.cast(0.,self.dtype), tf.stack([x,m,v])
    
    def minimize(self,func, t, state, par=1.):
        
        mask = tf.cast(par,self.dtype)
        x, m, v = state[0], state[1], state[2]

        with tf.GradientTape() as tape:
            tape.watch(x)
            f = func(x)
        g = tape.gradient(f,x) * mask
        
        t += 1.
        m = self.beta_1 * m + (1.-self.beta_1) * g
        v = self.beta_2 * v + (1.-self.beta_2) * tf.pow(g,2.0)
        mm = m / (1. - tf.pow(self.beta_1,t))
        vv = v / (1. - tf.pow(self.beta_2,t))

        x = x - self.lr * mm / (tf.sqrt(vv) + self.epsilon)
                
        return t, tf.stack([x,m,v])
        
class survival_permanental_process:
    
    # CONSTRUCTOR
    ##########################################################
    def __init__(self, kernel='Gaussian', eq_kernel='RFM',
                 eq_kernel_options={}):
        
        self.kernel, self.eq_kernel = kernel, eq_kernel
                        
        self.eq_kernel_options = {'n_dp':500,'n_rfm':100}
        for k,v in eq_kernel_options.items():
            self.eq_kernel_options[k] = v
        
        # Check the existence of the specified options
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        set_kernel    = ['Gaussian']
        set_eq_kernel = ['RFM']

        flag = False
        if kernel not in set_kernel:
            flag = True
            print('\nError in '+name+':\n'+
                  '   kernel ' + "\'"+kernel+"\' does not exist.\n")
        if eq_kernel not in set_eq_kernel:
            flag = True
            print('\nError in '+name+':\n'+
                  '   eq_kernel ' + "\'"+eq_kernel+"\' does not exist.\n")
        if flag:
            sys.exit()
        
    # FIT MODEL TO DATA 
    ##########################################################
    def fit(self, formula, df, set_par, lr=0.05, display=True, integral='Rectangle', id='id'):
        # df = [ID, start_time, stop_time, event, cov1, cov2,..]

        ok = False
        if formula.find('Surv(') != -1:
            s = formula.find('Surv(') + 5
            z = formula[s:s+formula[s:].find(')')].split(',')
            if len(z) == 3:
                [t0, t1, event] = [x.strip() for x in z]
                if formula.find('~') != -1:
                    c = [x.strip() for x in
                         formula[formula.find('~')+1:].split('+')]
                    ok = True
                    if integral == 'Simpson':
                        df = df.sort_values([id,t0])
        if not ok:
            print('\nError in '+name+':\n'+
                  "   invalid expression of 'formula'.\n")
            sys.exit()
        
        c_spk = df[df[event]==1][c].to_numpy()
        obs_r = self.integrate(df,t0,t1,id,integral)
        d_cov = df[c].to_numpy()
                
        self.c_spk, obs_region, d_cov = \
            np.array(c_spk,TYP), np.array(obs_r,TYP), np.array(d_cov,TYP)
        set_par = np.array(set_par,TYP)
        dim_cov = d_cov.shape[1]
        area =  sum(obs_region)
        value_set, est_set, par_set = [], [], []
        
        # Display the condition
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        if display:
            print("\n")
            print( "#################################################\n"
                   +"Survival Permanental Process\n"
                   +"Kernel:     "+self.kernel+"\n"
                   +"Eq. kernel: "+self.eq_kernel+"\n"
                   +"Cov. dim:   "+str(dim_cov)+"\n"
                   +"Event num:  "+str(len(c_spk))+"\n"
                   +"Obs. num:   "+str(len(obs_r))+"\n")
            
        elapse_t0 = time.time()
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        # Generate covariate samples
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        self.covariate(dim_cov)
        
        # Optimization option for v
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        m_rate = np.array(len(self.c_spk)/area, TYP)
        lr_v = lr
        eps_v = 1.e-5
        
        elapse_t1 = time.time()
                                        
        # Construct computation graph
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        if display:
            print("\rConstructing computation graph ...",end='')
        
        v_dammy = tf.constant(np.ones((len(self.c_spk),1),dtype=TYP))
        
        ml_abbrev = lambda x,init,y,g=False: \
            self.marginal_likelihood(x,self.c_spk,obs_region,d_cov,
                                     self.spk_sample,self.rfm_sample,
                                     init,eps_v,lr_v,y,g)
        # graph for function
        _ = ml_abbrev(set_par[0],v_dammy,False,g=True)
                        
        elapse_t2 = time.time()
        if display:
            print("\rConstructing computation graph ... finished!\n")
                
        # Initial value of v
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        init_v = 1./tf.sqrt(m_rate) * tf.cast(tf.ones([len(self.c_spk),1]),TYP)
        
        # Find the optimal hyper-parameter with grid search ^^^^^
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        for ii, par in enumerate(set_par):
            m, v, conv = ml_abbrev(par,init_v,display)[1:]
            #if not conv and len(set_par) != 1:
            #    continue
            value_set.append(m.numpy())
            par_set.append(par)
            est_set.append(v)
        
        indx = np.argmax(value_set)
        opt_value, opt_par = value_set[indx], tf.Variable(set_par[indx])
        self.est_v = est_set[indx]
        
        #opt_value, self.est_v = ml_abbrev(opt_par,init_v,False)[1:3]
        opt_par = opt_par.numpy()
        
        # Construct equivalent kernel for optimal hyper-parameter
        self.H = eval('self.eq_kernel_'+self.eq_kernel)\
            (self.kernel,opt_par,obs_region,d_cov,self.spk_sample,self.rfm_sample)
        self.H.compute_gram(self.c_spk)
        _ = self.H.compute_func_determinant(self.est_v)
        
        # Display the result 
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        elapse_t3 = time.time()
        el_graph, el_main = elapse_t2 - elapse_t1, elapse_t3 - elapse_t2 + elapse_t1 - elapse_t0
        if display:
            print("")
            print('Max Evidence: {0:.3e}'.format(opt_value))
            print('Hyper-param:  ['+', '.join(['{0:.3f}'.format(x) for x in opt_par])+']')
            print('Elapse time:  {0:.2f} for estimation, {1:.2f} for graph [sec]'\
                      .format(el_main,el_graph))
            print("################################################\n\n")
        
        return el_main 
    
    # CALCULATE MARGINAL LIKELIHOOD
    ##########################################################
    @tf.function(reduce_retracing=True, input_signature=[
            tf.TensorSpec(shape=[None],      dtype=TYP),
            tf.TensorSpec(shape=[None,None], dtype=TYP),
            tf.TensorSpec(shape=[None],      dtype=TYP),
            tf.TensorSpec(shape=[None,None], dtype=TYP),
            tf.TensorSpec(shape=[None,None], dtype=TYP),
            tf.TensorSpec(shape=[None,None], dtype=TYP),
            tf.TensorSpec(shape=[None,None], dtype=TYP),
            tf.TensorSpec(shape=None,        dtype=TYP),
            tf.TensorSpec(shape=None,        dtype=TYP),
            tf.TensorSpec(shape=None,        dtype=tf.bool),
            tf.TensorSpec(shape=None,        dtype=tf.bool),])
    def marginal_likelihood(self,par,c_spk,obs_region,d_cov,spk_sample,
                            rfm_sample,init_v,eps,lr,display=True,graph=False):
        
        if graph:
            n_ite = tf.constant(2)
            init_v = init_v[:1]
            c_spk = c_spk[:1]
        else:
            n_ite = tf.constant(1000)
        
        # Construct equivalent kernel ^^^^^^^^^^^^^^^^^^^^^^^
        H = eval('self.eq_kernel_'+self.eq_kernel)\
            (self.kernel,par,obs_region,d_cov,spk_sample,rfm_sample)
        H.compute_gram(c_spk)
        
        # Loss function  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        def loss(v):
            y = 2*tf.multiply(v,H.dot_gram(v)) - 1.
            return tf.reduce_mean(tf.square(y))
                        
        # Map estimation ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        opt = optimizer_adam(lr=lr)
        count, state = opt.reset(loss,init_v)
        i, conv = 0, False
        while i < n_ite - 1:
            count, state = opt.minimize(loss,count,state)
            min_value = loss(state[0])
            if (i+1)%20 == 0 and display:
                a = tf.strings.as_string(i+1,width=3)
                b = tf.strings.as_string(min_value,precision=2,scientific=True)
                t_pri = '\r#ite: '+a+', loss: '+b
                tf.print(t_pri, end='')
            if min_value < eps:
                conv = True
                break
            i += 1
        a = tf.strings.as_string(i+1,width=3)
        b = tf.strings.as_string(loss(state[0]),precision=2,scientific=True)
        t_pri = '\r#ite: '+a+', loss: '+b
        
        # Calculate marginal likelihood ^^^^^^^^^^^^^^^^^^^^^
        est_v = tf.squeeze(state[0],1)
        with tf.GradientTape() as tape2:
            tape2.watch(est_v)
            func_det_term = 0.5 * H.compute_func_determinant(est_v)
            action_int_term = - tf.reduce_sum(2*tf.math.log(tf.abs(est_v)) + 1.)
            marg_l = action_int_term + func_det_term
        if display:
            t_pri = t_pri + ', evidence: '+\
                tf.strings.as_string(marg_l,precision=4,scientific=True)
            tf.print(t_pri)
        
        dv_marg_l = tape2.gradient(marg_l, est_v)
        
        # Derivative of est_v regarding hyper-parameter ^^^^^
        H_v = H.dot_gram(tf.expand_dims(est_v,1))
        vv = - tf.squeeze(H.solve_Z_H(H_v))

        # Derivative of zz regarding hyper-parameter is 
        # equal to derivative of marginal likelihood
        correct = tf.reduce_sum(tf.multiply(tf.stop_gradient(dv_marg_l),vv))
        zz = marg_l + correct

        return -zz, marg_l, est_v, conv
                
    # PREDICT INTENSITY FUNCTION  
    ##########################################################
    def predict(self,y,conf_int = [0.025,0.5,0.975]):
        y = np.array(y).astype(TYP)
        y = y[:,np.newaxis] if len(y.shape) == 1 else y
        
        # Reduce large memory requirement by mini-batch
        n_max = 10**7
        nn = max(1 , n_max//self.c_spk.shape[0])
        batch_t = list(chunks(y,nn))
        mean_x = 2*self.H.dot(np.array(batch_t[-1]),self.c_spk,self.est_v[:,np.newaxis])
        var_x  = self.H.compute_covariance(np.array(batch_t[-1]))
        for tt in batch_t[-2::-1]:
            mean_x = tf.concat([2*self.H.dot(np.array(tt),self.c_spk,self.est_v[:,np.newaxis]),mean_x],0)
            var_x  = tf.concat([self.H.compute_covariance(tt),var_x],0)
        mean_x = tf.squeeze(mean_x,1)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        m, v = mean_x.numpy(), var_x.numpy()
        shape = (m**2+v)**2 / (2*v*(2*m**2+v))
        scale = 2*v*(2*m**2+v) / (m**2+v)
        est_map = scale * shape
        
        result = [gamma.ppf(ci,shape,scale=scale) for ci in conf_int]
        return result
    
    # GENERATE COVARIATE SAMPLES
    ##########################################################
    def covariate(self, dim_cov, seed=0):
                
        # Select data points for degenerate Nystrom approach
        self.spk_sample = [[]]
        if self.eq_kernel == 'Nystrom':
            n_dp = self.eq_kernel_options['n_dp']
            rng = default_rng(0)
            self.spk_sample = self.c_spk if n_dp >= len(self.c_spk) \
                else rng.choice(self.c_spk,size=n_dp,replace=False).astype(TYP)
        
        # Select random feature map for degenerate RFM approach
        self.rfm_sample = [[]]
        if self.eq_kernel == 'RFM':
            self.rfm_sample = eval(self.kernel+'_characteristic')\
                (self.eq_kernel_options['n_rfm']//2,dim_cov)

    # COMPUTE WEIGHTS OF NUMERICAL INTEGRATION OF SURVIVAL FUNCTION
    ##########################################################
    def integrate(self, df, t0, t1, id, integral):
        
        if integral == 'Simpson':
            weight = np.array([])
            for idd in df[id].unique():
                z = df[df[id]==idd][[t0,t1]].to_numpy()
                n = len(z)
                dt = (z[-1,1]-z[0,0])/n
                w = dt/3.0 * np.array([5]+[2,4]*(n//2-1)+[1])
                weight = np.r_[weight,w]
                if n%2 != 0:
                    print('\nError in '+name+':\n'+
                          "   #intervals in a unit must be even for Simpson's rule.\n")
                    sys.exit()
        else:
            weight = np.diff(df[[t0,t1]].to_numpy())[:,0]
        
        return weight
    
    # EQUIVALENT KERNEL WITH NAIVE APPROACH 
    ##########################################################
    class eq_kernel_Naive:
        
        def __init__(self,kernel,par,obs_region,d_cov,spk_sample,rfm_sample):
            self.p = d_cov
            self.kernel_func = lambda t,s,tr=False: eval(kernel+'_kernel')(t,s,par,tr)
            K = self.kernel_func(self.p,self.p)
            self.c = tf.cast(tf.linalg.diag(1./(2*obs_region)),TYP)
            self.chol = tf.linalg.cholesky(self.c + K)
            
        def compute_gram(self,x):
            self.spk = x
            K = self.kernel_func(x,x)
            k_ = self.kernel_func(self.p,x)
            k_chol_k = tf.matmul(k_,tf.linalg.cholesky_solve(self.chol, k_),transpose_a=True)
            self.H = K - k_chol_k
            
        def dot_gram(self,X):
            return tf.matmul(self.H,X)

        def dot(self,x,y,V):
            K = self.kernel_func(x,y)
            kx, ky = self.kernel_func(self.p,x), self.kernel_func(self.p,y)
            k_chol_k = tf.matmul(kx,tf.linalg.cholesky_solve(self.chol, ky),transpose_a=True)
            h = K - k_chol_k
            return tf.matmul(h,V)

        def trace(self,x):
            K = self.kernel_func(x,x,tr=True)
            k_ = self.kernel_func(self.p,x)
            k_chol_k = tf.reduce_sum(tf.multiply(k_,tf.linalg.cholesky_solve(self.chol, k_)),0)
            return K - k_chol_k
        
        def compute_func_determinant(self,est_v):
            # log|H| - log|K| - log|Z+H| + log|Z|
            Z_H = self.H + tf.linalg.diag(0.5/tf.square(est_v))
            self.chol_zh = tf.linalg.cholesky(Z_H)
            
            log_H_K = - tf.reduce_sum(2*tf.math.log(tf.linalg.diag_part(self.chol)))
            log_Z_H = tf.reduce_sum(2*tf.math.log(tf.linalg.diag_part(self.chol_zh)))
            log_Z = tf.reduce_sum(tf.math.log(0.5/tf.square(est_v)))
            
            return log_H_K - log_Z_H + log_Z
        
        def solve_Z_H(self,v):
            chol_zh = tf.stop_gradient(self.chol_zh)
                        
            return tf.linalg.cholesky_solve(chol_zh,v)
        
        def compute_covariance(self,x,trace=True):
            if trace:
                K = self.kernel_func(self.spk,x)
                kx, ky = self.kernel_func(self.p,self.spk), self.kernel_func(self.p,x)
                k_chol_k = tf.matmul(kx,tf.linalg.cholesky_solve(self.chol, ky),transpose_a=True)
                h_ = K - k_chol_k
                Z_H_h = tf.linalg.cholesky_solve(self.chol_zh, h_)
                h_Z_H_h = tf.reduce_sum(tf.multiply(h_,Z_H_h),0)
                
                return self.trace(x) - h_Z_H_h
            else:
                K = self.kernel_func(x,x)
                k_ = self.kernel_func(self.p,x)
                k_chol_k = tf.matmul(k_,tf.linalg.cholesky_solve(self.chol, k_),transpose_a=True)
                H = K - k_chol_k
                
                K = self.kernel_func(self.spk,x)
                kx, ky = self.kernel_func(self.p,self.spk), self.kernel_func(self.p,x)
                k_chol_k = tf.matmul(kx,tf.linalg.cholesky_solve(self.chol, ky),transpose_a=True)
                h_ = K - k_chol_k
                Z_H_h = tf.linalg.cholesky_solve(self.chol_zh, h_)
                h_Z_H_h = tf.matmul(h_,Z_H_h,transpose_a=True)
                
                return H - h_Z_H_h
    
    # EQUIVALENT KERNEL WITH DEGENERATE NYSTROM METHOD 
    ##########################################################
    class eq_kernel_Nystrom:
        
        def __init__(self,kernel,par,obs_region,d_cov,spk_sample,rfm_sample):
                        
            n_p = tf.shape(spk_sample)[0]
            self.kernel_func = lambda s,tr=False: eval(kernel+'_kernel')(spk_sample,s,par,tr)
            K = self.kernel_func(spk_sample)
            ei, ve = tf.linalg.eigh(K)
            ei, ve = ei[::-1], ve[:,::-1]
            index = tf.stop_gradient(tf.where(ei/ei[0]>1.e-6)[-1][0])
            ei, ve = ei[:index+1], ve[:,:index+1]
            # k(t,s) = <phi(t),phi(s)>, phi(t) = k_(t).T * ve
            ve = ve / tf.tile(tf.expand_dims(tf.sqrt(ei),0),(n_p,1))
            n_ei = tf.shape(ei)[0]

            self.c = tf.tile(tf.expand_dims(tf.sqrt(tf.cast(2*obs_region,TYP)),1),(1,n_ei))
            #^^^^^^^^^^            
            k_ = tf.transpose(self.kernel_func(d_cov))
            phi1 = tf.expand_dims(tf.multiply(tf.matmul(k_,ve),self.c),2)
            phi2 = tf.transpose(phi1,(0,2,1))
            xi = tf.reduce_sum(tf.matmul(phi1,phi2),0)
            #^^^^^^^^^^
            
            self.chol = tf.linalg.cholesky(xi + tf.eye(n_ei,dtype=TYP))
            x = tf.linalg.cholesky_solve(self.chol, self.chol)
            # h(t,s) = <psi(t),psi(s)>, psi(t) = self.vec * k_(t)
            
            vec = tf.transpose(tf.matmul(ve,x))
            self.psi = lambda t: tf.matmul(vec,self.kernel_func(t))
        
        def compute_gram(self,x):
            # H = R^t * R
            self.spk = x
            self.R = self.psi(x)
                                    
        def dot_gram(self,X):
            return tf.matmul(self.R,tf.matmul(self.R,X),transpose_a=True)

        def dot(self,x,y,V):
            rx, ry = self.psi(x), self.psi(y)
            return tf.matmul(rx,tf.matmul(ry,V),transpose_a=True)
        
        def trace(self,x):
            r_ = self.psi(x)
            return tf.reduce_sum(tf.multiply(r_,r_),0)
                    
        def compute_func_determinant(self,est_v):
            # log|H| - log|K| - log|Z+H| + log|Z|
            self.z_inv = tf.expand_dims(2*tf.square(est_v),0)
            Z_inv = tf.tile( self.z_inv, (tf.shape(self.R)[0],1) )
            Z_H_Z = tf.matmul(tf.multiply(self.R,Z_inv),self.R,transpose_b=True) \
                + tf.eye(tf.shape(self.R)[0],dtype=TYP)
            self.chol_zh = tf.linalg.cholesky(Z_H_Z)
            
            log_H_K = - tf.reduce_sum(2*tf.math.log(tf.linalg.diag_part(self.chol)))
            log_Z_H_Z = tf.reduce_sum(2*tf.math.log(tf.linalg.diag_part(self.chol_zh)))
            
            return log_H_K - log_Z_H_Z

        def solve_Z_H(self,v):
            R, chol_zh = tf.stop_gradient(self.R), tf.stop_gradient(self.chol_zh)
            z_inv = tf.transpose(self.z_inv)
            Z_inv_v = tf.multiply(z_inv,v)
            R_Z_inv_v = tf.matmul(R, Z_inv_v)
            y = tf.linalg.cholesky_solve(chol_zh, R_Z_inv_v)
            y = tf.multiply(z_inv,tf.matmul(R,y,transpose_a=True))
            
            return Z_inv_v - y
        
        def compute_covariance(self,x,trace=True):
            rx, rn = self.psi(x), self.psi(self.spk)
            if trace:
                # h(x,x)
                term0 = tf.reduce_sum(tf.multiply(rx,rx),0)
            
                # h(x).T * Z^{-1} * h(x)
                Z_inv = tf.tile( self.z_inv, (tf.shape(rn)[0],1) )
                rn_Z_inv = tf.multiply(rn,Z_inv)
                rn_Z_inv_rn = tf.matmul(rn_Z_inv,rn,transpose_b=True)
                term1 = tf.reduce_sum(tf.multiply(rx,tf.matmul(rn_Z_inv_rn,rx)),0)
                
                # h(x).T * Z^{-1} * R.T (I+R*Z^{-1}*R.T)^{-1} * R * Z^{-1} * h(x)  
                rn_Z_inv_R = tf.matmul(rn_Z_inv,self.R,transpose_b=True)
                v = tf.linalg.cholesky_solve(self.chol_zh, tf.transpose(rn_Z_inv_R))
                y = tf.matmul(tf.matmul(rn_Z_inv_R,v),rx)
                term2 = tf.reduce_sum(tf.multiply(rx,y),0)
            else:
                # h(x,x)
                term0 = tf.matmul(rx,rx,transpose_a=True)

                # h(x).T * Z^{-1} * h(x)
                Z_inv = tf.tile( self.z_inv, (tf.shape(rn)[0],1) )
                rn_Z_inv = tf.multiply(rn,Z_inv)
                rn_Z_inv_rn = tf.matmul(rn_Z_inv,rn,transpose_b=True)
                term1 = tf.matmul(rx,tf.matmul(rn_Z_inv_rn,rx),transpose_a=True)
                
                # h(x).T * Z^{-1} * R.T (I+R*Z^{-1}*R.T)^{-1} * R * Z^{-1} * h(x)  
                rn_Z_inv_R = tf.matmul(rn_Z_inv,self.R,transpose_b=True)
                v = tf.linalg.cholesky_solve(self.chol_zh, tf.transpose(rn_Z_inv_R))
                y = tf.matmul(tf.matmul(rn_Z_inv_R,v),rx)
                term2 = tf.matmul(rx,y,transpose_a=True)
                
            return term0 - term1 + term2

    # EQUIVALENT KERNEL WHITH DEGENERATE RANDOM FEATURE MAP
    ##########################################################
    class eq_kernel_RFM:
        
        def __init__(self,kernel,par,obs_region,d_cov,spk_sample,rfm_sample):
            n_rfm = 2*tf.shape(rfm_sample)[0]
            rfm_func = lambda t: eval(kernel+'_rfm')(rfm_sample,t,par)
            self.c = tf.tile(tf.expand_dims(tf.sqrt(tf.cast(2*obs_region,TYP)),0),(n_rfm,1))
            """
            #^^^^^^^^^^
            phi1 = tf.expand_dims(tf.transpose(tf.multiply(self.c,rfm_func(d_cov))),2)
            phi2 = tf.transpose(phi1,(0,2,1))
            xi = tf.reduce_sum(tf.matmul(phi1,phi2),0)
            #^^^^^^^^^^
            """
            """
            #^^^^^^^^^^
            tp = tf.float32
            phi1 = tf.expand_dims(tf.transpose(tf.multiply(self.c,rfm_func(d_cov))),2)
            xi = tf.cast(tf.zeros([n_rfm,n_rfm]),tp)
            for p1 in tf.split(tf.cast(phi1,tp),100):
                p2 = tf.transpose(p1,(0,2,1))
                xi = xi + tf.reduce_sum(tf.matmul(p1,p2),0)
            
            #^^^^^^^^^^
            """
            #^^^^^^^^^^
            phi1 = tf.multiply(self.c,rfm_func(d_cov))
            xi = tf.matmul(phi1,phi1,transpose_b=True)
                        
            self.chol = tf.linalg.cholesky(xi + tf.eye(n_rfm,dtype=TYP))
            x = tf.linalg.cholesky_solve(self.chol, self.chol)
            # h(t,s) = <psi(t),psi(s)>, psi(t) = x.T * k_(t)
            self.psi = lambda t: tf.matmul(x,rfm_func(t),transpose_a=True)
            
        def compute_gram(self,x):
            # H = R^t * R
            self.spk = x
            self.R = self.psi(x)
                        
        def dot_gram(self,X):
            return tf.matmul(self.R,tf.matmul(self.R,X),transpose_a=True)

        def dot(self,x,y,V):
            rx, ry = self.psi(x), self.psi(y)
            return tf.matmul(rx,tf.matmul(ry,V),transpose_a=True)
        
        def trace(self,x):
            r_ = self.psi(x)
            return tf.reduce_sum(tf.multiply(r_,r_),0)
        
        def compute_func_determinant(self,est_v):
            # log_func_det = log|H| - log|K| - log|Z+H| + log|Z|
            self.z_inv = tf.expand_dims(2*tf.square(est_v),0)
            Z_inv = tf.tile( self.z_inv, (tf.shape(self.R)[0],1) )
            Z_H_Z = tf.matmul(tf.multiply(self.R,Z_inv),self.R,transpose_b=True) \
                + tf.eye(tf.shape(self.R)[0],dtype=TYP)
            self.chol_zh = tf.linalg.cholesky(Z_H_Z)
            
            log_H_K = - tf.reduce_sum(2*tf.math.log(tf.linalg.diag_part(self.chol)))
            log_Z_H_Z = tf.reduce_sum(2*tf.math.log(tf.linalg.diag_part(self.chol_zh)))
            log_func_det = log_H_K - log_Z_H_Z
            
            return log_func_det
        
        def solve_Z_H(self,v):
            R, chol_zh = tf.stop_gradient(self.R), tf.stop_gradient(self.chol_zh)
            z_inv = tf.transpose(self.z_inv)
            Z_inv_v = tf.multiply(z_inv,v)
            R_Z_inv_v = tf.matmul(R, Z_inv_v)
            y = tf.linalg.cholesky_solve(chol_zh, R_Z_inv_v)
            y = tf.multiply(z_inv,tf.matmul(R,y,transpose_a=True))
            
            return Z_inv_v - y
        
        def compute_covariance(self,x,trace=True):
            rx, rn = self.psi(x), self.psi(self.spk)
            if trace:
                # h(x,x)
                term0 = tf.reduce_sum(tf.multiply(rx,rx),0)
            
                # h(x).T * Z^{-1} * h(x)
                Z_inv = tf.tile( self.z_inv, (tf.shape(rn)[0],1) )
                rn_Z_inv = tf.multiply(rn,Z_inv)
                rn_Z_inv_rn = tf.matmul(rn_Z_inv,rn,transpose_b=True)
                term1 = tf.reduce_sum(tf.multiply(rx,tf.matmul(rn_Z_inv_rn,rx)),0)
            
                # h(x).T * Z^{-1} * R.T (I+R*Z^{-1}*R.T)^{-1} * R * Z^{-1} * h(x)  
                rn_Z_inv_R = tf.matmul(rn_Z_inv,self.R,transpose_b=True)
                v = tf.linalg.cholesky_solve(self.chol_zh, tf.transpose(rn_Z_inv_R))
                y = tf.matmul(tf.matmul(rn_Z_inv_R,v),rx)
                term2 = tf.reduce_sum(tf.multiply(rx,y),0)
            else:
                # h(x,x)
                term0 = tf.matmul(rx,rx,transpose_a=True)
            
                # h(x).T * Z^{-1} * h(x)
                Z_inv = tf.tile( self.z_inv, (tf.shape(rn)[0],1) )
                rn_Z_inv = tf.multiply(rn,Z_inv)
                rn_Z_inv_rn = tf.matmul(rn_Z_inv,rn,transpose_b=True)
                term1 = tf.matmul(rx,tf.matmul(rn_Z_inv_rn,rx),transpose_a=True)
            
                # h(x).T * Z^{-1} * R.T (I+R*Z^{-1}*R.T)^{-1} * R * Z^{-1} * h(x)  
                rn_Z_inv_R = tf.matmul(rn_Z_inv,self.R,transpose_b=True)
                v = tf.linalg.cholesky_solve(self.chol_zh, tf.transpose(rn_Z_inv_R))
                y = tf.matmul(tf.matmul(rn_Z_inv_R,v),rx)
                term2 = tf.matmul(rx,y,transpose_a=True)
            
            return term0 - term1 + term2

# KERNEL FUNCTIONS
##########################################################
@tf.function(input_signature=(tf.TensorSpec(shape=[None,None], dtype=TYP),
                              tf.TensorSpec(shape=[None,None], dtype=TYP),
                              tf.TensorSpec(shape=[None], dtype=TYP),
                              tf.TensorSpec(shape=None, dtype=tf.bool),))
def Gaussian_kernel(t,s,par,trace=False):
    par = tf.cast(par,t.dtype)
    a, b, = par[0], par[1:]
    if trace == True:
        bb = tf.tile(tf.expand_dims(b,axis=0),(tf.shape(t)[0],1))
        y = tf.exp( -tf.pow(bb*(t-t),2.) )
        # [len(t)]
        return a * tf.reduce_prod(y,axis=1)
    else:
        tt = tf.tile(tf.expand_dims(tf.transpose(t),axis=2),[1,1,tf.shape(s)[0]])
        ss = tf.tile(tf.expand_dims(tf.transpose(s),axis=2),[1,1,tf.shape(t)[0]])
        ss = tf.transpose(ss,perm=[0,2,1])
        bb = tf.tile(tf.expand_dims(tf.expand_dims(b,axis=1),axis=2),(1,tf.shape(t)[0],tf.shape(s)[0]))
        y = tf.exp( -tf.pow(bb*(tt-ss),2.) )
        # [len(t), len(s)]
        return a * tf.reduce_prod(y,axis=0)

def Gaussian_characteristic(n_rfm,n_dim,seed=0):
    rng = default_rng(seed)
    return rng.normal(loc=0.,scale=1.,size=(n_rfm,n_dim)).astype(TYP)

def Gaussian_rfm(rfm,t,par):
    
    par, rfm = tf.cast(par,t.dtype), tf.cast(rfm,t.dtype)
    a, b, n = par[0], par[1:], tf.cast(tf.shape(rfm)[0],t.dtype)
    w = tf.concat([rfm,rfm],axis=0)
    d0 = tf.zeros((tf.shape(rfm)[0],tf.shape(t)[0]),dtype=t.dtype)
    d1 = -0.5*np.pi*tf.ones((tf.shape(rfm)[0],tf.shape(t)[0]),dtype=t.dtype)
    d = tf.concat([d0,d1],axis=0)
    
    # [len(omega), len(t)]
    bb = tf.tile(tf.expand_dims(b,axis=0),(tf.shape(w)[0],1))
    phase = tf.matmul(tf.sqrt(2*bb**2)*w,t,transpose_b=True) + d
    return tf.cos(phase) * tf.sqrt(a/n)
        
