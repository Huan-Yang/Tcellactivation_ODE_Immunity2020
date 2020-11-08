import numpy as np
import pandas as pd
from multiprocessing import Pool, Lock
from scipy.optimize import curve_fit, least_squares


def run_curve_fit(pars):
    f, x, y, yerr, p0, bnds, log = pars
    try:
        if np.any(yerr == 0):
            popt, pcov = curve_fit(f, x, y, p0=p0, bounds=bnds)
        else:
            popt, pcov = curve_fit(f, x, y, p0=p0, bounds=bnds, sigma=yerr)
        sol = f(x, *popt)
        err = np.sum(np.power(sol - y, 2))
        if log is not None:
            flog = open(log, 'a')
            flog.write('\n' + '\t'.join(str(p) for p in popt))
            flog.write('\t' + '\t'.join(str(p) for p in p0))
            flog.write('\t{}'.format(err))
            flog.close()
        return [*popt, *p0, err]
    except RuntimeError:
        print('could not fit with p0 = [{}]'.format(
            ','.join([str(p) for p in p0])))
        return None


def run_curve_fit_mc(pars):
    f, x, y, yerr, p0, bnds, log = pars
    try:
        if np.any(yerr == 0):
            popt, pcov = curve_fit(f, x, y, p0=p0, bounds=bnds)
        else:
            popt, pcov = curve_fit(f, x, y, p0=p0, bounds=bnds, sigma=yerr)
        sol = f(x, *popt)
        err = np.sum(np.power(sol - y, 2))
        if log is not None:
            lock.acquire()
            flog = open(log, 'a')
            flog.write('\n' + '\t'.join(str(p) for p in popt))
            flog.write('\t' + '\t'.join(str(p) for p in p0))
            flog.write('\t{}'.format(err))
            flog.close()
            lock.release()
        return [*popt, *p0, err]
    except RuntimeError:
        print('could not fit with p0 = [{}]'.format(
            ','.join([str(p) for p in p0])))
        return None


def init(l):
    global lock
    lock = l


def fit_pars(f, xdata, ydata, yerr, p0_list, parnames, bounds=(-np.inf, np.inf), ncores=25, fn_log=None):
    if fn_log is not None:
        flog = open(fn_log, 'w')
        flog.write('\t'.join(parnames))
        flog.write('\t' + '\t'.join('{}0'.format(pn) for pn in parnames))
        flog.write('\terr')
        flog.close()
    # run fitting
    df_pars = pd.DataFrame(
        columns=parnames + ['{}0'.format(pn) for pn in parnames] + ['err'])
    if ncores > 1:
        l = Lock()
        p = Pool(processes=ncores, initializer=init, initargs=(l,))
        out = p.map(run_curve_fit_mc, [
                    [f, xdata, ydata, yerr, p0, bounds, fn_log] for p0 in p0_list])
        for row in out:
            if row is None:
                continue
            df_pars = df_pars.append(pd.DataFrame(
                [[*row]], columns=df_pars.columns))
        p.terminate()
    else:
        for p0 in p0_list:
            row = run_curve_fit([f, xdata, ydata, yerr, p0, bounds, fn_log])
            if row is None:
                continue
            df_pars = df_pars.append(pd.DataFrame(
                [[*row]], columns=df_pars.columns))
    return df_pars


def fit_pars_lsq(f_res, p0_list, parnames, bounds=(-np.inf, np.inf), ncores=25, fn_log=None):
    if fn_log is not None:
        flog = open(fn_log, 'w')
        flog.write('\t'.join(parnames))
        flog.write('\t' + '\t'.join('{}0'.format(pn) for pn in parnames))
        flog.write('\tcost\tstatus')
        flog.close()
    # run fitting
    # df_pars = pd.DataFrame(columns=parnames + ['{}0'.format(pn) for pn in parnames] + ['err', 'status'])
    df_pars = None
    if ncores > 1:
        l = Lock()
        p = Pool(processes=ncores, initializer=init, initargs=(l,))
        out = p.map(run_lsq, [[f_res, p0, bounds, fn_log] for p0 in p0_list])
        for row in out:
            if row is None:
                continue
            if df_pars is None:
                rescols = ['r{}'.format(i) for i in range(
                    1, len(row)-2*len(parnames)-1)]
                df_pars = pd.DataFrame(columns=parnames + ['{}0'.format(pn) for pn in parnames]
                                       + ['err', 'status']+rescols)
            df_pars = df_pars.append(pd.DataFrame(
                [[*row]], columns=df_pars.columns))
        p.terminate()
    else:
        for p0 in p0_list:
            row = run_lsq([f_res, p0, bounds, fn_log])
            if row is None:
                continue
            if df_pars is None:
                rescols = ['r{}'.format(i) for i in range(
                    1, len(row)-2*len(parnames)-1)]
                df_pars = pd.DataFrame(columns=parnames + ['{}0'.format(pn) for pn in parnames]
                                       + ['err', 'status']+rescols)
            df_pars = df_pars.append(pd.DataFrame(
                [[*row]], columns=df_pars.columns))
    return df_pars


def run_lsq(pars):
    f_res, p0, bnds, log = pars
    try:
        res = least_squares(f_res, p0, bounds=bnds)
        if log is not None:
            lock.acquire()
            flog = open(log, 'a')
            flog.write('\n' + '\t'.join(str(p) for p in res.x))
            flog.write('\t' + '\t'.join(str(p) for p in p0))
            flog.write('\t{}\t{}'.format(res.cost, res.status))
            flog.close()
            lock.release()
        return [*res.x, *p0, res.cost, res.status, *res.fun]
    except RuntimeError:
        print('could not fit with p0 = [{}]'.format(
            ','.join([str(p) for p in p0])))
        return None
