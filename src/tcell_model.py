import copy
from collections import namedtuple
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import pchip
import pandas as pd
from prettytable import PrettyTable

State = namedtuple('State', ['Uu', 'Us', 'Dneg', 'Dpos', 'IL2',
                             'CD25u', 'CD25neg', 'CD25pos',
                             'CD80neg', 'CD80pos', 'CD80u',
                             'CD86neg', 'CD86pos', 'CD86u'])


class Parameters(object):
    __slots__ = ('p_U', 'p_D', 'p_D_cd25', 'p_Dneg', 'p_Dpos', 'p_D_cd28', 'd_U', 'd_Uu', 'd_Us',
                 'k_D_cd28', 'k_D_ctla4', 'k_D_il2', 'k_D_cd25', 'k_Dpos', 'k_Dpos_CD25', 'k_Dneg', 'k_Dneg_CD25', 'k_U', 'k_Dpos_ctla4',
                 'n_D_cd28', 'n_D_ctla4', 'n_D_il2', 'n_D_cd25', 'n_Dpos', 'n_Dpos_CD25', 'n_Dneg', 'n_Dneg_CD25', 'n_U', 'n_Dpos_ctla4',
                 'U_0', 'Dneg_0', 'Dpos_0', 'IL2_0', 'delay', 'f_CD25_p', 'f_CD25_Dneg', 'f_unsens', 'f', 'fu', 'fs')
    default_value = 0

    def __init__(self, *args, **kwargs):
        # Set default values
        for att in self.__slots__:
            if att.startswith('n_'):
                setattr(self, att, 2)
            else:
                setattr(self, att, self.default_value)

        # Set attributes passed in as arguments
        for k, v in zip(self.__slots__, args):
            setattr(self, k, v)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __iter__(self):
        for attr in self.__slots__:
            yield attr, getattr(self, attr)

    def print_table(self, parnames=None):
        tab = PrettyTable(field_names=['name', 'value'])
        if parnames is None:
            parnames = self.__slots__
        for pn in parnames:
            tab.add_row([pn, '{:.2e}'.format(getattr(self, pn))])
        print(tab)

    def get_names(self):
        return self.__slots__

    def get_pd_series(self):
        return pd.Series([getattr(self, pn) for pn in self.__slots__], index=self.__slots__)

    def __str__(self):
        return ",".join(["{}={}".format(attr, getattr(self, attr)) for attr in self.__slots__])


def popt_to_pars(popt, parnames, default):
    pars = copy.deepcopy(default)
    for i, pn in enumerate(parnames):
        if pn in pars.get_names():
            setattr(pars, pn, popt[i])
    return pars


def dfrow_to_pars(row, default):
    pars = copy.deepcopy(default)
    for pn in pars.get_names():
        if pn not in row.columns:
            continue
        setattr(pars, pn, row[pn].values[0])
    return pars


class ActivationModel:

    def __init__(self, activ, Uu_death, Us_death, Dneg_growth, Dpos_growth,
                 CTLA4_on, CTLA4_off, delay=0):
        self.activ = activ
        self.Uu_death = Uu_death
        self.Us_death = Us_death
        self.Dneg_growth = Dneg_growth
        self.Dpos_growth = Dpos_growth
        self.CTLA4_on = CTLA4_on
        self.CTLA4_off = CTLA4_off
        self.delay = delay

    def _get_interpolation(self, df):
        if df.index[0] == 0:
            return pchip(df.index, df.values)
        else:
            return pchip(np.concatenate(([0], df.index)), np.concatenate(([0], df.values)))

    def run(self, y0, t, expdata, method='RK45', rtol=1e-3, atol=1e-6):
        if len(y0) != 4:
            return None
        inputs = ['CD25u', 'CD25neg', 'CD25pos', 'CD80pos', 'CD80neg', 'CD86pos', 'CD86neg',
                  'CD80u', 'CD86u', 'IL2']
        for name in inputs:
            if name in expdata:
                setattr(self, name, self._get_interpolation(expdata[name]))
            else:
                setattr(self, name, pchip([0, 1], [1, 1]))
        t = np.array(t)
        soln = solve_ivp(self.ode, [0, t[-1]], y0, t_eval=t,
                         method=method, rtol=rtol, atol=atol)
        self.success = soln.success
        self.state = State(Uu=soln.y[0, :], Us=soln.y[1, :], Dneg=soln.y[2, :], Dpos=soln.y[3, :],
                           IL2=self.IL2(t), CD25u=self.CD25u(t), CD25neg=self.CD25neg(t), CD25pos=self.CD25pos(t),
                           CD80neg=self.CD80neg(t), CD80pos=self.CD80pos(t), CD80u=self.CD80u(t),
                           CD86neg=self.CD86neg(t), CD86pos=self.CD86pos(t), CD86u=self.CD86u(t))
        self.time = t

    def ode(self, t, y):
        state = State(Uu=y[0], Us=y[1], Dneg=y[2], Dpos=y[3], IL2=self.IL2(t),
                      CD25u=self.CD25u(t), CD25neg=self.CD25neg(t), CD25pos=self.CD25pos(t),
                      CD80neg=self.CD80neg(t), CD80pos=self.CD80pos(t), CD80u=self.CD80u(t),
                      CD86neg=self.CD86neg(t), CD86pos=self.CD86pos(t), CD86u=self.CD80u(t))
        if t < self.delay:
            dUu = -self.Uu_death(state) * state.Uu
            dUs = -self.Us_death(state) * state.Us
            dDneg = (self.Dneg_growth(state) - self.CTLA4_on(state)) * \
                state.Dneg + self.CTLA4_off(state) * state.Dpos
        else:
            dUu = -self.Uu_death(state) * state.Uu
            dUs = -(self.Us_death(state) + self.activ(state)) * state.Us
            dDneg = (self.Dneg_growth(state) - self.CTLA4_on(state)) * state.Dneg + self.CTLA4_off(
                state) * state.Dpos + 2 * self.activ(state) * state.Us
        dDpos = (self.Dpos_growth(state) - self.CTLA4_off(state)) * \
            state.Dpos + self.CTLA4_on(state) * state.Dneg
        return [dUu, dUs, dDneg, dDpos]

    def get_df(self):
        return pd.DataFrame({'time': self.time,
                             'undivided unresp': self.state.Uu,
                             'undivided resp': self.state.Us,
                             'undivided': self.state.Uu+self.state.Us,
                             'divided CTLA4-': self.state.Dneg,
                             'divided CTLA4+': self.state.Dpos,
                             'cell count': self.state.Uu + self.state.Us + self.state.Dneg + self.state.Dpos,
                             'divided count': self.state.Dneg + self.state.Dpos,
                             'IL2': self.state.IL2,
                             'CD25 U': self.state.CD25u,
                             'CD25 CTLA4-': self.state.CD25neg,
                             'CD25 CTLA4+': self.state.CD25pos,
                             'U division rate': self.activ(self.state),
                             'Us death rate': self.Us_death(self.state),
                             'CTLA4- growth rate': self.Dneg_growth(self.state),
                             'CTLA4+ growth rate': self.Dpos_growth(self.state),
                             'CTLA4- -> CTLA4+ rate': self.CTLA4_on(self.state),
                             'Uu division flux': self.activ(self.state) * self.state.Uu,
                             'Us division flux': self.activ(self.state) * self.state.Us,
                             'CTLA4- growth flux': self.Dneg_growth(self.state) * self.state.Dneg,
                             'CTLA4+ growth flux': self.Dpos_growth(self.state) * self.state.Dpos,
                             'CTLA4- -> CTLA4+ flux': self.CTLA4_on(self.state) * self.state.Dneg,
                             'CD80neg': self.state.CD80neg, 'CD80pos': self.state.CD80pos, 'CD80u': self.state.CD80u,
                             'CD86neg': self.state.CD86neg, 'CD86pos': self.state.CD86pos, 'CD86u': self.state.CD86u})


def _hillf(s, n, k):
    if s == 0:
        return 0
    else:
        return (s ** n) / (k ** n + s ** n)


def _hilla(s, n, k):
    h = np.zeros_like(s)
    h[s > 0] = np.power(s[s > 0], n) / (k ** n + np.power(s[s > 0], n))
    return h


def hill(s, n, k):
    if not isinstance(s, (np.ndarray, list, tuple)):
        return _hillf(s, n, k)
    else:
        return _hilla(s, n, k)
