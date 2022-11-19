import os
import datetime

import numpy as np
import scipy
import pandas as pd
import pandas.tseries.holiday

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sn

import wallstreet as ws

import ipywidgets as widgets
from IPython.display import clear_output

class BinomialOption:
    params = ['s','k','t','sigma','r','type','tree','style']
    valid_types = {
        'c':'C',
        'C':'C',
        'Call':'C',
        'call':'C',
        'p':'P',
        'P':'P',
        'Put':'P',
        'put':'P'
    }

    valid_styles = {
        'a':'A',
        'A':'A',
        'American':'A',
        'american':'A',
        'e':'E',
        'E':'E',
        'European':'E',
        'european':'E',
    }

    def __init__(self, s, k, t, sigma, r, type: str='C', style: str='A', n: int=50, qty: int = 1):
        self.s = s
        self.k = k
        if isinstance(t,str) or isinstance(t,datetime.date) or isinstance(t,datetime.datetime):
            self.t = self.date_to_t(t)
        else:
            self.t = t
        self.sigma = sigma
        self.r = r
        self.n = n
        self.qty = qty
        self.pos = 'long' if qty > 0 else 'short'

        if type not in self.valid_types.keys():
            raise ValueError('`type` must be \'call\', \'C\', \'put\', or \'P\'')
        else:
            self.type = self.valid_types.get(type)
        if style not in self.valid_styles.keys():
            raise ValueError('`style` must be \'American\', \'A\', \'European\', or \'E\'')
        else:
            self.style = self.valid_styles.get(style)

        self.get_secondary_params()

        self.default_params = {param:self.__dict__.get(param) for param in self.params}


    def __repr__(self):
        return f'BinomialOption(s={self.s}, k={self.k}, t={self.t}, sigma={self.sigma}, r={self.r}, type={self.type}, style={self.style})'

    def __neg__(self):
        return BinomialOption(self.s, self.k, self.t, self.sigma, self.r, type = self.type, style=self.style, n=self.n,qty=-self.qty)

    def __add__(self,other):
        return OptionPortfolio(self,other)

    def __sub__(self,other):
        return OptionPortfolio(self,-other)

    def __mul__(self,amount: int):
        return BinomialOption(self.s, self.k, self.t, self.sigma, self.r, type=self.type, style=self.style, n=self.n, qty=self.qty*amount)


    @staticmethod
    def date_to_t(date):
        if isinstance(date,str):
            date = pd.to_datetime(date).date()
        elif isinstance(date,datetime.datetime):
            date = date.date()

        today = pd.Timestamp.today()
        us_holidays = pd.tseries.holiday.USFederalHolidayCalendar()
        holidays = us_holidays.holidays(start=today, end=date)
        dt = len(pd.bdate_range(start=today, end=date)) - len(holidays)
        
        return dt/252

    def reset_params(self):
        for param in self.params:
            self.__dict__[param] = self.default_params[param]
            self.get_secondary_params()
            if hasattr(self,'val_tree'):
                delattr(self,'val_tree')
    
    def get_secondary_params(self):
        self.dt = self.t / self.n
        self.r_hat = (1+self.r) ** self.dt
        self.up = np.exp(self.sigma*np.sqrt(self.dt))
        self.dn = 1/self.up
        self.pi = (self.r_hat - self.dn)/(self.up - self.dn)
        self.tree = self.create_tree()

    def summary(self):
        data = {
                '':['price','delta','gamma','vega',''],
                ' ':[self.value(),self.delta(),self.gamma(),self.vega(),'']
                }
        df = pd.DataFrame(data)
        df2 = pd.DataFrame({' ':['S','K','IV','t','r'],'':[self.s,self.k,self.sigma,self.t,self.r]})

        summary_df = pd.concat({'greeks':df,'parameters':df2},axis=1)
        return summary_df

    def evaluate(self,price):
        val = max(price-self.k,0) if self.type == 'C' else max(self.k-price,0)
        return val

    def node_eval(self,price,Vu,Vd):
        intrinsic_value = price - self.k if self.type == 'C' else self.k - price
        if self.style == 'A':
            val = max(intrinsic_value, (1/self.r_hat) * ((self.pi*Vu)+((1-self.pi)*Vd)))
        else:
            val = (1/self.r_hat) * ((self.pi*Vu)+((1-self.pi)*Vd))
        return val


    def create_tree(self):
        tree = [[0]]
        for period in range(self.n):
            layer = []
            for node in tree[-1]:
                layer += [node+1,node-1]
            layer = sorted(list(set(layer)),reverse=True)
            tree.append(layer)
        tree = [[self.s*(self.up**node) for node in layer] for layer in tree]
        return tree

    def create_val_tree(self,val_tree=None):
        if val_tree is None:
            val_tree = [[self.evaluate(node) for node in self.tree[-1]]]

        tree_idx = -len(val_tree)-1
        layer = []
        for i, price in enumerate(self.tree[tree_idx]):
            Vu = val_tree[-1][i]
            Vd = val_tree[-1][i+1]
            layer.append(self.node_eval(price,Vu,Vd))

        val_tree.append(layer)
        
        if len(val_tree[-1]) != 1:
            self.create_val_tree(val_tree)

        self.val_tree = val_tree

    def value(self,**kwargs):
        if kwargs:
            for key, val in kwargs.items():
                self.__dict__[key] = val
                self.get_secondary_params()
                self.create_val_tree()

        if not hasattr(self,'val_tree'):
            self.create_val_tree()
        
        result = self.val_tree[-1][0]

        if kwargs:
            self.reset_params()

        return self.qty*result

    def price(self):
        return self.value()

    @property
    def premium(self):
        return self.value()

    def delta(self,**kwargs):
        if kwargs:
            for key, val in kwargs.items():
                self.__dict__[key] = val
                self.get_secondary_params()
                self.create_val_tree()
        elif not hasattr(self,'val_tree'):
            self.create_val_tree()

        layer = self.val_tree[-2]
        result = (layer[0]-layer[1])/(self.s*(self.up-self.dn))

        if kwargs:
            self.reset_params()

        return self.qty*result

    def gamma(self,precision=1e-4):
        result =  (self.delta(s=self.s+precision) - self.delta(s=self.s-precision))/(2*precision)
        return self.qty*result

    def vega(self,precision=1e-4):
        result =  (self.value(sigma=self.sigma+precision) - self.value(sigma=self.sigma-precision))/(100*precision)
        return self.qty*result

    def theta(self,precision=1e-4):
        '''Poor approximation of theta, avoid using'''
        result =  -(self.value(t=self.t+precision) - self.value(t=self.t-precision))/(2*precision)
        return self.qty*result / 365

    def rho(self,precision=1e-4):
        '''Poor approximation of rho, avoid using'''
        result =  (self.value(r=self.r+precision) - self.value(r=self.r-precision))/(2*precision)
        return self.qty*result / 100

    def plot_dist(self):
        sn.kdeplot(np.concatenate(self.tree),fill=True)

    def show_tree(self):
        x = []
        for i in range(len(self.tree)):
            x += [i]*len(self.tree[i])

        ys = []
        for i in self.tree:
            ys += i

        plt.plot(x,ys,'o',markersize=1)
        plt.show()

class BSOption:
    params = ['s','k','t','sigma','r','type']
    valid_types = {
        'c':'C',
        'C':'C',
        'Call':'C',
        'call':'C',
        'p':'P',
        'P':'P',
        'Put':'P',
        'put':'P'
    }

    def __init__(self,s=100,k=100,t=1,sigma=0.3,r=0.035,type='C',qty=1):
        self.s = s
        self.k = k
        if isinstance(t,str) or isinstance(t,datetime.date) or isinstance(t,datetime.datetime):
            self.t = self.date_to_t(t)
        else:
            self.t = t
        self.sigma = sigma
        self.r = r
        self.qty = qty
        self.pos = 'long' if qty > 0 else 'short'
        if type not in self.valid_types.keys():
            raise ValueError('`type` must be \'call\', \'C\', \'put\', or \'P\'')
        else:
            self.type = self.valid_types.get(type)

        self.price = self.value
        self.default_params = {param:self.__dict__.get(param) for param in self.params}
        self.norm_cdf = scipy.stats.norm.cdf
        self.deriv = scipy.misc.derivative

    def __neg__(self):
        return BSOption(self.s, self.k, self.t, self.sigma, self.r, type=self.type, qty=-self.qty)

    def __add__(self,other):
        return OptionPortfolio(self,other)

    def __sub__(self,other):
        return OptionPortfolio(self,-other)

    def __mul__(self,amount: int):
        return BSOption(self.s, self.k, self.t, self.sigma, self.r, type=self.type, qty=self.qty*amount)

    def __repr__(self):
        sign = '+' if self.qty > 0 else ''
        return f'{sign}{self.qty} BSOption(s={self.s}, k={self.k}, t={self.t}, sigma={self.sigma}, r={self.r}, type={self.type})'

    def reset_params(self):
        for param in self.params:
            self.__dict__[param] = self.default_params[param]

    @staticmethod
    def date_to_t(date):
        if isinstance(date,str):
            date = pd.to_datetime(date).date()
        elif isinstance(date,datetime.datetime):
            date = date.date()

        today = pd.Timestamp.today()
        us_holidays = pd.tseries.holiday.USFederalHolidayCalendar()
        holidays = us_holidays.holidays(start=today, end=date)
        dt = len(pd.bdate_range(start=today, end=date)) - len(holidays)
        
        return dt/252


    def d1(self):
        return (np.log(self.s/(self.k*((1+self.r)**-self.t))) + ((0.5*self.sigma**2))*self.t)/(self.sigma*(self.t**0.5))

    def d2(self):
        return self.d1() - self.sigma*np.sqrt(self.t)
    
    def value(self,**kwargs):
        for key, val in kwargs.items():
            self.__dict__[key] = val
        
        if self.type == 'C':
            result = self.s*self.norm_cdf(self.d1()) - self.k*((1+self.r)**-self.t)*self.norm_cdf(self.d2())
        elif self.type == 'P':
            result = self.k*((1+self.r)**-self.t)*self.norm_cdf(-self.d2()) - self.s*self.norm_cdf(-self.d1())

        if kwargs:
            self.reset_params()

        return self.qty*result
    
    @property
    def premium(self):
        return self.value()

    def delta(self,**kwargs):
        for key, val in kwargs.items():
            self.__dict__[key] = val
        
        result = self.norm_cdf(self.d1())
        if self.type == 'P':
            result -= 1

        if kwargs:
            self.reset_params()

        return self.qty*result
        
    
    def gamma(self,**kwargs):
        for key, val in kwargs.items():
            self.__dict__[key] = val

        result = np.exp(-(self.d1())**2/2)/np.sqrt(2*np.pi)/(self.s*self.sigma*np.sqrt(self.t))

        if kwargs:
            self.reset_params()

        return self.qty*result

    def vega(self,**kwargs):
        for key, val in kwargs.items():
            self.__dict__[key] = val

        result = self.s*np.exp(-(self.d1())**2/2)/np.sqrt(2*np.pi)*np.sqrt(self.t)/100

        if kwargs:
            self.reset_params()

        return self.qty*result

    def theta(self,**kwargs):
        for key, val in kwargs.items():
            self.__dict__[key] = val

        if self.type == 'C':
            result = -self.s * scipy.stats.norm.pdf(self.d1()) * self.sigma / (2 * np.sqrt(self.t)) - self.r * self.k * np.exp(-self.r * self.t) * scipy.stats.norm.cdf(self.d2())
        elif self.type == 'P':
            result = -self.s * scipy.stats.norm.pdf(self.d1()) * self.sigma / (2 * np.sqrt(self.t)) + self.r * self.k * np.exp(-self.r * self.t) * scipy.stats.norm.cdf(-self.d2())

        result = result/365

        return self.qty*result

    def rho(self,**kwargs):
        for key, val in kwargs.items():
            self.__dict__[key] = val

        if self.type == 'C':
            result = self.k * self.t * np.exp(-self.r * self.t) * scipy.stats.norm.cdf(self.d2())
        elif self.type == 'P':
            result = -self.k * self.t * np.exp(-self.r * self.t) * scipy.stats.norm.cdf(-self.d2())

        if kwargs:
            self.reset_params()

        return self.qty*result / 100

    def values(self):
        spot = np.linspace(self.k*0.85,self.k*1.15,80)
        vals = np.array([self.value(s=i,t=1e-6) for i in spot])
        return vals

    def summary(self):
        data = {
                '':['price','delta','gamma','vega','theta','rho'],
                ' ':[self.price(),self.delta(),self.gamma(),self.vega(),self.theta(),self.rho()]
                }
        df = pd.DataFrame(data)
        df2 = pd.DataFrame({' ':['S','K','IV','t','r',''],'':[self.s,self.k,self.sigma,self.t,self.r,'']})

        summary_df = pd.concat({'parameters':df2,'characteristics / greeks':df},axis=1)
        return summary_df

    def plot(self,var='payoff', interactive=False, resolution=40, return_ax=False):
        '''`var` must be either \'value\', \'delta\', \'gamma\', \'vega\', \'theta\', \'rho\', \'payoff\', or \'pnl\''''
        greeks = {'value','delta','gamma','vega','rho','theta','pnl','payoff'}

        if var not in greeks: 
            raise ValueError('`var` must be either \'value\', \'delta\', \'gamma\', \'vega\', \'theta\', \'rho\', \'payoff\', or \'pnl\'')

        spot = np.linspace(self.k*0.66,self.k*1.33,resolution)

        if not interactive or var=='payoff':
            if var == 'payoff':
                vals = np.array([self.value(s=i,t=1e-6) for i in spot])
            elif var == 'pnl':
                cost = self.value()
                vals = np.array([self.value(s=i,t=1e-6) - cost for i in spot])
            elif var == 'value':
                vals = np.array([self.value(s=i) for i in spot])
            elif var == 'delta':
                vals = np.array([self.delta(s=i) for i in spot])
            elif var == 'gamma':
                vals = np.array([self.gamma(s=i) for i in spot])
            elif var == 'vega':
                vals = np.array([self.vega(s=i) for i in spot])
            elif var == 'theta':
                vals = np.array([self.theta(s=i) for i in spot])
            elif var == 'rho':
                vals = np.array([self.rho(s=i) for i in spot])

            plt.plot(spot,vals)
            plt.axhline(0,color='black')
            plt.axvline(self.k,linestyle='--',color='gray',alpha=0.7)

            if return_ax:
                return plt.gca()
            else:
                plt.show()
        else:
            def f(t=2,k=100,sigma=0.1,r=0.04):
                kwargs = {'t':t,'k':k,'t':t,'sigma':sigma,'r':r}
                if var == 'payoff':
                    plt.plot(spot,[self.value(s=i,**kwargs) for i in spot],label='Value')
                    plt.plot(spot,[self.value(s=i,k=k,r=r,sigma=sigma,t=1e-6) for i in spot],label='Payoff at Expiration')
                elif var == 'pnl':
                    cost = self.value()
                    plt.plot(spot,[self.value(s=i,**kwargs) - cost for i in spot],label='Value')
                    plt.plot(spot,[self.value(s=i,k=k,r=r,sigma=sigma,t=1e-6) - cost for i in spot],label='Payoff at Expiration')
                elif var == 'value':
                    plt.plot(spot,[self.value(s=i,**kwargs) for i in spot],label='Value')
                elif var == 'delta':
                    plt.plot(spot,[self.delta(s=i,**kwargs) for i in spot],label='$\Delta$')
                elif var == 'gamma':
                    plt.plot(spot,[self.gamma(s=i,**kwargs) for i in spot],label='$\Gamma$')
                elif var == 'vega':
                    plt.plot(spot,[self.vega(s=i,**kwargs) for i in spot],label='Vega')
                elif var == 'theta':
                    plt.plot(spot,[self.theta(s=i,**kwargs) for i in spot],label='$\Theta$')
                elif var == 'rho':
                    plt.plot(spot,[self.rho(s=i,**kwargs) for i in spot],label='Rho')
                    
                plt.axhline(0,color='black')
                plt.axvline(k,linestyle='--',color='gray',alpha=0.7)
                plt.legend()
                plt.show()

            interactive_plot = widgets.interactive(f, t=(0.001,2.0,0.001),sigma=(0.01,1.0,0.01), r = (0.0,0.08,0.0025), k = (self.k*0.8,self.k*1.2,0.1))
            output = interactive_plot.children[-1]
            output.layout.height = '450px'
            return interactive_plot

class OptionPortfolio:

    def __init__(self,*args):
        self.options = args

    def __repr__(self):
        os = '\n'.join([repr(o) for o in self.options])
        output = f'OptionPortfolio(\n{os}\n)'
        return output

    def __add__(self,other):
        return OptionPortfolio(*(list(self.options) + [other]))

    def __sub__(self,other):
        return OptionPortfolio(*(list(self.options) + [-other]))

    def value(self, **kwargs):
        return sum(i.value(**kwargs) for i in self.options)

    def price(self):
        return self.value()

    @property
    def premium(self):
        return self.value()

    def delta(self, **kwargs):
        return sum(i.delta(**kwargs) for i in self.options)

    def gamma(self, **kwargs):
        return sum(i.gamma(**kwargs) for i in self.options)

    def vega(self, **kwargs):
        return sum(i.vega(**kwargs) for i in self.options)

    def theta(self, **kwargs):
        return sum(i.theta(**kwargs) for i in self.options)

    def rho(self, **kwargs):
        return sum(i.rho(**kwargs) for i in self.options)

    def summary(self):
        data = {
                '':['total cost','delta','gamma','vega','theta','rho'],
                ' ':[self.price(),self.delta(),self.gamma(),self.vega(),self.theta(),self.rho()]
                }
        df = pd.DataFrame(data)
        dat = {'':['price','S','K','IV','t','r']}
        format_qty = lambda x: f'+{x}' if x>0 else f'{x}'
        dat.update({f'Leg {idx+1} ({format_qty(o.qty)}{o.type})':[o.price(),o.s,o.k,o.sigma,o.t,o.r] for idx, o in enumerate(self.options)})
        df2 = pd.DataFrame(dat)

        summary_df = pd.concat({'parameters':df2,'characteristics / greeks':df},axis=1)
        return summary_df

    def plot(self,var='payoff', interactive=False, resolution=40, return_ax=False):
        '''`var` must be either \'value\', \'delta\', \'gamma\', \'vega\', \'theta\', \'rho\', \'payoff\', or \'pnl\''''
        greeks = {'value','delta','gamma','vega','rho','theta','pnl','payoff'}

        if var not in greeks: 
            raise ValueError('`var` must be either \'value\', \'delta\', \'gamma\', \'vega\', \'theta\', \'rho\', \'payoff\', or \'pnl\'')

        ks = [o.k for o in self.options]
        spot = np.linspace(min(ks)*0.66,max(ks)*1.33,resolution)

        if not interactive or var=='payoff':
            if var == 'payoff':
                vals = np.array([self.value(s=i,t=1e-6) for i in spot])
            elif var == 'pnl':
                cost = self.value()
                vals = np.array([self.value(s=i,t=1e-6) - cost for i in spot])
            elif var == 'value':
                vals = np.array([self.value(s=i) for i in spot])
            elif var == 'delta':
                vals = np.array([self.delta(s=i) for i in spot])
            elif var == 'gamma':
                vals = np.array([self.gamma(s=i) for i in spot])
            elif var == 'vega':
                vals = np.array([self.vega(s=i) for i in spot])
            elif var == 'theta':
                vals = np.array([self.theta(s=i) for i in spot])
            elif var == 'rho':
                vals = np.array([self.rho(s=i) for i in spot])

            plt.plot(spot,vals)
            plt.axhline(0,color='black')
            for k in ks:
                plt.axvline(k,linestyle='--',color='gray',alpha=0.7)

            if return_ax:
                return plt.gca()
            else:
                plt.show()
        else:
            def f(t=2):
                if var == 'payoff':
                    plt.plot(spot,[self.value(s=i,t=t) for i in spot],label='Value')
                    plt.plot(spot,[self.value(s=i,t=1e-6) for i in spot],label='Payoff at Expiration')
                elif var == 'pnl':
                    cost = self.value()
                    plt.plot(spot,[self.value(s=i,t=t) - cost for i in spot],label='Value')
                    plt.plot(spot,[self.value(s=i,t=1e-6) - cost for i in spot],label='Payoff at Expiration')
                elif var == 'value':
                    plt.plot(spot,[self.value(s=i,t=t) for i in spot],label='Value')
                elif var == 'delta':
                    plt.plot(spot,[self.delta(s=i,t=t) for i in spot],label='$\Delta$')
                elif var == 'gamma':
                    plt.plot(spot,[self.gamma(s=i,t=t) for i in spot],label='$\Gamma$')
                elif var == 'vega':
                    plt.plot(spot,[self.vega(s=i,t=t) for i in spot],label='Vega')
                elif var == 'theta':
                    plt.plot(spot,[self.theta(s=i,t=t) for i in spot],label='$\Theta$')
                elif var == 'rho':
                    plt.plot(spot,[self.rho(s=i,t=t) for i in spot],label='Rho')
                    
                plt.axhline(0,color='black')
                for k in ks:
                    plt.axvline(k,linestyle='--',color='gray',alpha=0.7)
                plt.legend()
                plt.show()

            interactive_plot = widgets.interactive(f, t=(0.001,2.0,0.001))
            output = interactive_plot.children[-1]
            output.layout.height = '450px'
            return interactive_plot


class VolSurface:

    def __init__(self, ticker, moneyness=False):
        self.ticker = ticker
        self.moneyness = moneyness

    def get_data(self):
        underlying = ws.Stock(self.ticker)
        self.spot = underlying.price
        call = ws.Call(self.ticker)
        dates = pd.to_datetime(call.expirations).sort_values()
        dates = [date for date in dates if date > datetime.date.today()]

        for date in dates:
            call = ws.Call(self.ticker,date.day,date.month,date.year)
            clear_output()
            put = ws.Put(self.ticker,date.day,date.month,date.year)
            clear_output()
            dat = pd.DataFrame(call.data + put.data)
            dat['date'] = date
            dat['type'] = dat.contractSymbol.str[-9]
            if date == dates[0]:
                data = dat

            data = pd.concat([data,dat])
        
        return data

    def get_vol_surface(self,moneyness=False):
        if not hasattr(self,'data'):
            self.data = self.get_data()

        vol_data = self.data[['strike','impliedVolatility','lastPrice','date','type']]
        vol_data = vol_data[((vol_data.strike >= self.spot)&(vol_data.type=='C'))|((vol_data.strike < self.spot)&(vol_data.type=='P'))]
        if moneyness:
            vol_data['strike'] = (vol_data.strike / self.spot)*100

        return vol_data.sort_values(['date','strike'])

    def skew_plot(self,*args):
        if not hasattr(self,'surface'):
            self.surface = self.get_vol_surface(moneyness=self.moneyness)

        idx = 0
        if args:
            idx = [int(i) for i in args if str(i).isnumeric()]

        tbl = self.surface.pivot_table('impliedVolatility','strike','date').dropna()
        tbl.iloc[:,idx].plot()
        ttl = tbl.columns[idx][0].strftime('Expiration: %m-%d-%Y') if idx!=0 else tbl.columns[idx].strftime('Expiration: %m-%d-%Y')
        plt.title(ttl)
        if self.moneyness:
            plt.xlabel('strike (% moneyness)')

    def surface_plot(self):
        if not hasattr(self,'surface'):
            self.surface = self.get_vol_surface(moneyness=self.moneyness)

        fig = go.Figure(data=[go.Mesh3d(x=self.surface.strike, y=self.surface.date, z=self.surface.impliedVolatility, intensity=self.surface.impliedVolatility)])
        fig.show()

    @property
    def surface_table(self):
        if not hasattr(self,'surface'):
            self.surface = self.get_vol_surface(moneyness=self.moneyness)
        return self.surface.pivot_table('impliedVolatility','strike','date').dropna()


if __name__=='__main__':
    pass