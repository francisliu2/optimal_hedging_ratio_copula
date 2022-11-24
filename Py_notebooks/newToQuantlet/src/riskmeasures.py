# from toolbox import *
#
# def ERM_estimate_trapezoidal(k, rh):
#     rh = np.sort(rh)
#     s  = ECDF(rh)(rh)
#     d  = s[1:] - s[:-1]
#     toint = ERM_weight(k,s)*rh
#     return -np.sum((toint[:-1] + toint[1:])*d)/2
#
# def ES(q, rh):
#     b = np.quantile(rh,q)
#     return -np.mean(rh[rh<=q])
#
# def wrapper(rs, rf, h, risk_measure):
#     rh = rs - h*rf
#     return risk_measure(rh)
#
# def optimize_h(C, k_arr, q_arr):
#     sample = C.sample(1000000)
#     rs = sample[:,0]
#     rf = sample[:,1]
#     best_h = []
#
#     for k in k_arr:
#         fn = lambda h: wrapper(rs,rf,h,partial(ERM_estimate_trapezoidal,k))
#         best_h.append(scipy.optimize.fmin(fn,1)[0])
#
#     for q in q_arr:
#         fn = lambda h: wrapper(rs,rf,h,partial(ES,q))
#         best_h.append(scipy.optimize.fmin(fn,1)[0])
#     return best_h