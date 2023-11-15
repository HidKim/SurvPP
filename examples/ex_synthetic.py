# This is the script that reproduces our result on synthetic data
# in (Kim, NeurIPS2023).

from pylab import *
from sklearn.model_selection import KFold
from sksurv.metrics import cumulative_dynamic_auc
import tensorflow as tf
from HidKim_SurvPP import survival_permanental_process as SPP
import dill

def main():
    # data type: ['linear', 'nonlinear']
    ty = 'nonlinear'
    # Num. of sub-regions for each individual: [1,5,10,20,50]
    J = 5
    # Evaluation points of performance: 0 ~ 1
    score_t = [0.3, 0.5, 0.7, 0.9]
    # Num. of cross validation
    n_split = 10
    
    # synthetic data you download from our github page 
    f_data = ty+'_U1000_J'+str(J).zfill(2)+'.dill'
    data = dill.load(open('data/synthetic/'+f_data,'rb'))
    
    # DATA SPLITTING #################################
    df, cov_func = data['df'], data['func']
    kf = KFold(n_splits=n_split, shuffle=True, random_state=0)
    df_train, df_test = [], []
    for indx_train, indx_test in kf.split(unique(df['id'].to_numpy())):
        df_train.append(df[df['id'].isin(indx_train)])
        df_test.append(df[df['id'].isin(indx_test)])

    # ESTIMATION & PREDICTION ########################
    score = {x:[] for x in ['auc','tll','cpu']}
    score['t'] = score_t
    for df_tr, df_te in zip(df_train,df_test):
        auc, tll, cpu = estimation_spp(df_tr,df_te,score_t,cov_func)
        score['auc'].append(auc)
        score['tll'].append(tll)
        score['cpu'].append(cpu)
    
    # BOX PLOT of RESULT #############################
    subplot(2,5,(1,4))
    posit, lw = range(len(score_t)), 1.0
    z = array(score['tll'])
    bp = boxplot(z,positions=posit,widths=0.2,patch_artist=True,notch=False,
                 boxprops={'facecolor':'r','linewidth':lw},
                 flierprops={'markersize':4, 'markerfacecolor':'r'},
                 medianprops={'color':'k','linewidth':lw},
                 whiskerprops={'linewidth':lw,'linestyle':'--'},
                 capprops={'linewidth':lw})
    ylim([-0.6,0.2])
    xticks(posit,[str(x) for x in score_t])
    title('TLL: J_u = '+str(J))

    subplot(2,5,(6,9))
    posit, lw = range(len(score_t)), 1.0
    z = array(score['auc'])
    bp = boxplot(z,positions=posit,widths=0.2,patch_artist=True,notch=False,
                 boxprops={'facecolor':'r','linewidth':lw},
                 flierprops={'markersize':4, 'markerfacecolor':'r'},
                 medianprops={'color':'k','linewidth':lw},
                 whiskerprops={'linewidth':lw,'linestyle':'--'},
                 capprops={'linewidth':lw})
    ylim([0.1,1.0])
    xticks(posit,[str(x) for x in score_t])
    title('AUC: J_u = '+str(J))

    subplot(2,5,(5,10))
    errorbar(0, mean(score['cpu']), yerr=std(score['cpu']), capsize=5,
             fmt='o', markersize=5, ecolor='r', markerfacecolor='r',
             markeredgecolor='r')
    xlim([-1,1])
    ylim([1.e-4,60])
    yscale('log')
    title('CPU [sec]')
    tight_layout()
    show()
    

def estimation_spp(df_tr, df_te, score_t, cov_func):
    
    # Shape data for evaluation
    surv_train = df_tr.groupby('id').max()[['event','t1']].to_numpy()
    surv_test  = df_te.groupby('id').max()[['event','t1']]
    list_id, surv_test = surv_test.index.values, surv_test.to_numpy()
    # Make structed array -> survival_train, survival_test
    survival_train = zeros(len(surv_train),dtype=[('event',bool),('t1',float64)])
    survival_test  = zeros(len(surv_test),dtype=[('event',bool),('t1',float64)])
    survival_train['event'], survival_train['t1'] = surv_train[:,0], surv_train[:,1]
    survival_test['event'],  survival_test['t1']  = surv_test[:,0], surv_test[:,1]

    # Estimation and prediction
    model = SPP(kernel='Gaussian', eq_kernel='RFM', eq_kernel_options={'n_rfm':500})
    with tf.device('/cpu:0'):
        set_par = [[1,x,x,x] for x in [0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0, 7.0, 10.0]]
        cpu = model.fit('Surv(t0,t1,event) ~ cov1 + cov2 + t1', df=df_tr, set_par=set_par)
        cpu = cpu / len(set_par)
                
    # Calculate cumulative hazard function and performances
    t = array(sorted(unique(list(linspace(0,1,1000))+\
                            list(surv_test[:,1])+list(score_t))))
    tt = 0.5*(t[1:]+t[:-1])
    risk_id, tll_id = [], []
    for id, event, t1 in zip(list_id,surv_test[:,0],surv_test[:,1]):
        [[a1,w1,b1],[a2,w2,b2]] = cov_func[id]
        cov1, cov2 = a1*cos(2*pi*w1*t+pi*b1), a2*cos(2*pi*w2*t+pi*b2)

        s = model.predict(c_[cov1,cov2,t],[0.5])[0]
        cum_haz = r_[0,cumsum(0.5*(s[:-1]+s[1:])*diff(t))]
        risk_id.append(cum_haz[isin(t,score_t)])
        
        s, e = minimum(score_t,t1), event * (t1<=score_t)
        cov1, cov2 = a1*cos(2*pi*w1*t1+pi*b1), a2*cos(2*pi*w2*t1+pi*b2)        
        tll = - array([cum_haz[where(t==ss)][0] for ss in s]) + \
            e*log(model.predict(array([[cov1,cov2,t1]]),[0.5])[0][0])
        tll_id.append(tll)
    
    # Performance score 
    auc, _ = cumulative_dynamic_auc(survival_train,survival_test,risk_id,score_t)
    tll = mean(tll_id,0)

    return auc, tll, cpu

if __name__ == "__main__":
    main()
