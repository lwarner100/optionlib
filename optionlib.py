import os
import datetime

import numpy as np
import scipy
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sn

import wallstreet as ws

import ipywidgets as widgets
from IPython.display import clear_output

class BinomialOption:
    params = ['s','k','t','sigma','r','type_','tree','style']
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

    valid_pos = {
        'l':'l',
        'long':'l',
        's':'s',
        'short':'s'
    }

    def __init__(self, s, k, t, sigma, r, type_: str='C', style: str='A', n: int=50, pos: str='l',qty: int = 1):
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

        if type_ not in self.valid_types.keys():
            raise ValueError('`type_` must be \'call\', \'C\', \'put\', or \'P\'')
        else:
            self.type_ = self.valid_types.get(type_)
        if style not in self.valid_styles.keys():
            raise ValueError('`style` must be \'American\', \'A\', \'European\', or \'E\'')
        else:
            self.style = self.valid_styles.get(style)
        if pos.lower() not in self.valid_pos.keys():
            raise ValueError('`pos` must be \'l\', \'long\', \'s\', or \'short\'')
        else:
            self.pos = self.valid_pos.get(pos.lower())

        self.get_secondary_params()

        self.default_params = {param:self.__dict__.get(param) for param in self.params}


    def __repr__(self):
        return f'BinomialOption(s={self.s}, k={self.k}, t={self.t}, sigma={self.sigma}, r={self.r}, type={self.type_}, style={self.style})'

    def __neg__(self):
        return BinomialOption(self.s, self.k, self.t, self.sigma, self.r, type_ = self.type_, style=self.style, n=self.n, pos='s',qty=self.qty)

    def __add__(self,other):
        return OptionPortfolio(self,other)

    def __sub__(self,other):
        return OptionPortfolio(self,-other)

    def __mul__(self,amount: int):
        return BinomialOption(self.s, self.k, self.t, self.sigma, self.r, type_=self.type_, style=self.style, n=self.n, pos='s', qty=self.qty*amount)


    def date_to_t(self,date):
        if isinstance(date,str):
            date = pd.to_datetime(date).date()
        elif isinstance(date,datetime.datetime):
            date = date.date()

        try:
            import pandas_market_calendars as mcal
            nyse = mcal.get_calendar('NYSE')
            days = nyse.schedule(start_date=datetime.date.today(), end_date=date)
            dt = len(days.index)/252
        except ImportError:
            dt = (date - datetime.date.today()).days/365
        
        return dt

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
        val = max(price-self.k,0) if self.type_ == 'C' else max(self.k-price,0)
        return val

    def node_eval(self,price,Vu,Vd):
        intrinsic_value = price - self.k if self.type_ == 'C' else self.k - price
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

        return self.qty*(result if self.pos=='l' else -result)

    def price(self):
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

        return self.qty*(result if self.pos=='l' else -result)

    def gamma(self,precision=1e-4):
        result =  (self.delta(s=self.s+precision) - self.delta(s=self.s-precision))/(2*precision)
        return self.qty*(result if self.pos=='l' else -result)

    def vega(self,precision=1e-4):
        result =  (self.value(sigma=self.sigma+precision) - self.value(sigma=self.sigma-precision))/(100*precision)
        return self.qty*(result if self.pos=='l' else -result)

    def theta(self,precision=1e-4):
        result =  -(self.value(t=self.t+precision) - self.value(t=self.t-precision))/(200*precision)
        return self.qty*(result if self.pos=='l' else -result)

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
    params = ['s','k','t','sigma','r','type_']
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

    valid_pos = {
        'l':'l',
        'long':'l',
        's':'s',
        'short':'s'
    }

    def __init__(self,s=100,k=100,t=1,sigma=0.3,r=0.035,type_='C',pos='l',qty=1):
        self.s = s
        self.k = k
        if isinstance(t,str) or isinstance(t,datetime.date) or isinstance(t,datetime.datetime):
            self.t = self.date_to_t(t)
        else:
            self.t = t
        self.sigma = sigma
        self.r = r
        self.qty = qty
        if type_ not in self.valid_types.keys():
            raise ValueError('`type_` must be \'call\', \'C\', \'put\', or \'P\'')
        else:
            self.type_ = self.valid_types.get(type_)
        if pos.lower() not in self.valid_pos.keys():
            raise ValueError('`pos` must be \'l\', \'long\', \'s\', or \'short\'')
        else:
            self.pos = self.valid_pos.get(pos.lower())
        self.price = self.value
        self.default_params = {param:self.__dict__.get(param) for param in self.params}
        self.norm_cdf = scipy.stats.norm.cdf
        self.deriv = scipy.misc.derivative

    def __neg__(self):
        return BSOption(self.s, self.k, self.t, self.sigma, self.r, type_=self.type_, pos='s',qty=self.qty)

    def __add__(self,other):
        return OptionPortfolio(self,other)

    def __sub__(self,other):
        return OptionPortfolio(self,-other)

    def __mul__(self,amount: int):
        return BSOption(self.s, self.k, self.t, self.sigma, self.r, type_=self.type_, qty=self.qty*amount)

    def __repr__(self):
        return f'BSOption(s={self.s}, k={self.k}, t={self.t}, sigma={self.sigma}, r={self.r}, type={self.type_})'

    def reset_params(self):
        for param in self.params:
            self.__dict__[param] = self.default_params[param]

    def date_to_t(self,date):
        if isinstance(date,str):
            date = pd.to_datetime(date).date()
        elif isinstance(date,datetime.datetime):
            date = date.date()

        try:
            import pandas_market_calendars as mcal
            nyse = mcal.get_calendar('NYSE')
            days = nyse.schedule(start_date=datetime.date.today(), end_date=date)
            dt = len(days.index)/252
        except:
            dt = (date - datetime.date.today()).days/365
        
        return dt


    def d1(self):
        return (np.log(self.s/(self.k*((1+self.r)**-self.t))) + ((0.5*self.sigma**2))*self.t)/(self.sigma*(self.t**0.5))

    def d2(self):
        return self.d1() - self.sigma*np.sqrt(self.t)
    
    def value(self,**kwargs):
        for key, val in kwargs.items():
            self.__dict__[key] = val
        
        if self.type_ == 'C':
            result = self.s*self.norm_cdf(self.d1()) - self.k*((1+self.r)**-self.t)*self.norm_cdf(self.d2())
        elif self.type_ == 'P':
            result = self.k*((1+self.r)**-self.t)*self.norm_cdf(-self.d2()) - self.s*self.norm_cdf(-self.d1())

        if kwargs:
            self.reset_params()

        return self.qty*(result if self.pos=='l' else -result)

    def delta(self,**kwargs):
        for key, val in kwargs.items():
            self.__dict__[key] = val
        
        result = self.norm_cdf(self.d1())
        if self.type_ == 'P':
            result -= 1

        if kwargs:
            self.reset_params()

        return self.qty*(result if self.pos=='l' else -result)
        
    
    def gamma(self,**kwargs):
        for key, val in kwargs.items():
            self.__dict__[key] = val

        result = np.exp(-(self.d1())**2/2)/np.sqrt(2*np.pi)/(self.s*self.sigma*np.sqrt(self.t))

        if kwargs:
            self.reset_params()

        return self.qty*(result if self.pos=='l' else -result)

    def vega(self,**kwargs):
        for key, val in kwargs.items():
            self.__dict__[key] = val

        result = self.s*np.exp(-(self.d1())**2/2)/np.sqrt(2*np.pi)*np.sqrt(self.t)/100

        if kwargs:
            self.reset_params()

        return self.qty*(result if self.pos=='l' else -result)

    def theta(self,**kwargs):
        for key, val in kwargs.items():
            self.__dict__[key] = val

        if self.type_ == 'C':
            result = -((self.s * np.exp((-self.d1()**2)/2 )/np.sqrt(2*np.pi) * self.sigma) / 2*(self.t**0.5)) - (self.r*self.k*np.exp(-self.t*(1+self.r))*self.norm_cdf(self.d2()))
        else:
            result = -((self.s * np.exp((-self.d1()**2)/2 )/np.sqrt(2*np.pi) * self.sigma) / 2*(self.t**0.5)) + (self.r*self.k*np.exp(-self.t*(1+self.r))*self.norm_cdf(-self.d2()))

        if kwargs:
            self.reset_params()

        return self.qty*(result if self.pos=='l' else -result)

    def values(self):
        spot = np.linspace(self.k*0.85,self.k*1.15,80)
        vals = np.array([self.value(s=i,t=1e-6) for i in spot])
        return vals

    def summary(self):
        data = {
                '':['price','delta','gamma','vega',''],
                ' ':[self.price(),self.delta(),self.gamma(),self.vega(),'']
                }
        df = pd.DataFrame(data)
        df2 = pd.DataFrame({' ':['S','K','IV','t','r'],'':[self.s,self.k,self.sigma,self.t,self.r]})

        summary_df = pd.concat({'greeks':df,'parameters':df2},axis=1)
        return summary_df
    
    def payoff_plot(self):
        spot = np.linspace(self.k*0.85,self.k*1.15,80)
        vals = np.array([self.value(s=i,t=1e-6) for i in spot])

        plt.plot(spot,vals)
        plt.vlines(self.k,-1,15,linestyle='--',alpha=0.5)
        plt.ylim(-1,self.k*0.15)
        plt.show()

    def pnl_plot(self):
        spot = np.linspace(self.k*0.85,self.k*1.15,80)
        vals = np.array([self.value(s=i,t=1e-6) for i in spot])
        price = self.value()

        plt.plot(spot,vals-price)
        plt.vlines(self.k,-1.2*price,15,linestyle='--',alpha=0.5)
        plt.hlines(0,self.k*0.85,self.k*1.15,color='black')
        plt.ylim(-1.2*price,self.k*0.15)
        plt.show()

    def interactive_plot(self,var: str = 'value'):
        '''`var` must be either \'value\', \'delta\', \'gamma\', or \'vega\''''
        greeks = {'value','delta','gamma','vega'}
        
        if var not in greeks: 
            raise ValueError('`var` must be either \'value\', \'delta\', \'gamma\', or \'vega\'')

        if var=='value':
            spot = np.linspace(self.s*0.85,self.s*1.15,80)

            def f(k=100,t=2,sigma=0.1,r=0.0075):
                plt.plot(spot,[self.value(s=i,k=k,t=t,sigma=sigma,r=r) for i in spot])
                plt.plot(spot,[self.value(s=i,k=k,sigma=sigma,t=1e-6,r=r) for i in spot])
                plt.vlines(k,-1,15,linestyle='--',alpha=0.5)
                plt.ylim(-1,self.s*0.15)
                plt.show()

        elif var=='delta':
            spot = np.linspace(self.s/2,self.s*1.5,50)

            def f(k=100,t=2,sigma=0.1,r=0.0075):
                plt.plot(spot,[self.delta(s=i,t=t,sigma=sigma,k=k,r=r) for i in spot],label='$\Delta$')
                if self.type_ =='C':
                    plt.vlines(k,0,1,linestyle='--',alpha=0.5)
                else:
                    plt.vlines(k,0,-1,linestyle='--',alpha=0.5)
                plt.legend()
                plt.show()
                self.reset_params()

        elif var=='gamma':
            spot = np.linspace(self.s*0.7,self.s*1.3,100)
            
            def f(k=100,t=2,sigma=0.1,r=0.0075):
                plt.plot(spot,[self.gamma(s=i,t=t,sigma=sigma,k=k,r=r) for i in spot],label='$\Gamma$')
                plt.vlines(k,0,1,linestyle='--',alpha=0.5)
                plt.ylim(0,0.5)
                plt.legend()
                plt.show()
                self.reset_params()

        elif var=='vega':
            spot = np.linspace(self.s*0.7,self.s*1.3,100)
            
            def f(k=100,t=2,sigma=0.1,r=0.0075):
                plt.plot(spot,[self.vega(s=i,t=t,sigma=sigma,k=k,r=r) for i in spot],label='Vega')
                plt.vlines(k,0,1,linestyle='--',alpha=0.5)
                plt.ylim(0,0.5)
                plt.legend()
                plt.show()
                self.reset_params()

        interactive_plot = widgets.interactive(f, t=(0.001,2.0,0.001),sigma=(0.01,1.0,0.01), r = (0.0,0.04,0.0001), k = (self.s*0.9,self.s*1.1,0.1))
        output = interactive_plot.children[-1]
        output.layout.height = '450px'
        return interactive_plot

class OptionPortfolio:

    def __init__(self,*args):
        self.options = args

    def __repr__(self):
        os = '\n'.join([f'+{o.qty} {o.__repr__()}' if o.pos=='l' else f'-{o.qty} {o.__repr__()}' for o in self.options])
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

    def delta(self, **kwargs):
        return sum(i.delta(**kwargs) for i in self.options)

    def gamma(self, **kwargs):
        return sum(i.gamma(**kwargs) for i in self.options)

    def vega(self, **kwargs):
        return sum(i.vega(**kwargs) for i in self.options)

    def summary(self):
        data = {
                '':['price','delta','gamma','vega',''],
                ' ':[self.price(),self.delta(),self.gamma(),self.vega(),'']
                }
        df = pd.DataFrame(data)
        dat = {'':['S','K','IV','t','r']}
        pos_symbol = lambda x: '+' if x=='l' else '-'
        dat.update({f'Leg {idx+1} ({pos_symbol(o.pos)}{o.type_})':[o.s,o.k,o.sigma,o.t,o.r] for idx, o in enumerate(self.options)})
        df2 = pd.DataFrame(dat)

        summary_df = pd.concat({'greeks':df,'parameters':df2},axis=1)
        return summary_df

    def plot(self,var='payoff', interactive=False, resolution=60):
        '''`var` must be either \'value\', \'delta\', \'gamma\', \'vega\',\'payoff\', or \'pnl\''''
        greeks = {'value','delta','gamma','vega','pnl','payoff'}

        if var not in greeks: 
            raise ValueError('`var` must be either \'value\', \'delta\', \'gamma\', or \'vega\'')

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

            plt.plot(spot,vals)
            plt.axhline(0,color='black')
            for k in ks:
                plt.axvline(k,linestyle='--',color='gray',alpha=0.7)
            plt.show()
        else:
            if var=='value':
                def f(t=2):
                    plt.plot(spot,[self.value(s=i,t=t) for i in spot],label='Value')
                    plt.plot(spot,[self.value(s=i,t=1e-6) for i in spot],label='Payoff at Expiration')
                    plt.axhline(0,color='black')
                    for k in ks:
                        plt.axvline(k,linestyle='--',color='gray',alpha=0.7)
                    plt.legend()
                    plt.show()

            if var=='pnl':
                cost = self.value()
                def f(t=2):
                    plt.plot(spot,[self.value(s=i,t=t) - cost for i in spot],label='Value')
                    plt.plot(spot,[self.value(s=i,t=1e-6) - cost for i in spot],label='Payoff at Expiration')
                    plt.axhline(0,color='black')
                    for k in ks:
                        plt.axvline(k,linestyle='--',color='gray',alpha=0.7)
                    plt.legend()
                    plt.show()

            elif var=='delta':
                def f(t=2):
                    plt.plot(spot,[self.delta(s=i,t=t) for i in spot],label='$\Delta$')
                    plt.axhline(0,color='black')
                    for k in ks:
                        plt.axvline(k,linestyle='--',color='gray',alpha=0.7)
                    plt.legend()
                    plt.show()

            elif var=='gamma':
                def f(t=2):
                    plt.plot(spot,[self.gamma(s=i,t=t) for i in spot],label='$\Gamma$')
                    plt.axhline(0,color='black')
                    for k in ks:
                        plt.axvline(k,linestyle='--',color='gray',alpha=0.7)
                    plt.legend()
                    plt.show()

            elif var=='vega':
                def f(t=2):
                    plt.plot(spot,[self.vega(s=i,t=t) for i in spot],label='Vega')
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
    call = BSOption()
    put = BSOption(type_='p')
    call.payoff_plot()