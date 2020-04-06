import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Leer datos reales: casos confirmados, muertes y recuperados
ConfirmedCases_raw=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
Deaths_raw=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
Recoveries_raw=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

### Preprocesamiento datasets en pandas
def cleandata(df_raw):
    df_cleaned=df_raw.melt(id_vars=['Province/State','Country/Region','Lat','Long'],value_name='Cases',var_name='Date')
    df_cleaned=df_cleaned.set_index(['Country/Region','Province/State','Date'])
    return df_cleaned 

ConfirmedCases=cleandata(ConfirmedCases_raw)
Deaths=cleandata(Deaths_raw)
Recoveries=cleandata(Recoveries_raw)

### Datos por país
def countrydata(df_cleaned,oldname,newname):
    df_country=df_cleaned.groupby(['Country/Region','Date'])['Cases'].sum().reset_index()
    df_country=df_country.set_index(['Country/Region','Date'])
    df_country.index=df_country.index.set_levels([df_country.index.levels[0], pd.to_datetime(df_country.index.levels[1])])
    df_country=df_country.sort_values(['Country/Region','Date'],ascending=True)
    df_country=df_country.rename(columns={oldname:newname})
    return df_country
  
ConfirmedCasesCountry=countrydata(ConfirmedCases,'Cases','Total')
DeathsCountry=countrydata(Deaths,'Cases','Total')
RecoveriesCountry=countrydata(Recoveries,'Cases','Total')

### Datos mundiales
def world_data(df_country):
    df_world = df_country.groupby(['Date'])['Total'].sum().reset_index()
    df_world = df_world.set_index(['Date'])
    df_world = df_world.sort_values(['Date'],ascending=True)
    return df_world

TotalCasesWorld = world_data(ConfirmedCasesCountry)
DeathsWorld = world_data(DeathsCountry)
RecoveriesWorld = world_data(RecoveriesCountry)

N = 10000		# Se usará una población pequeña para que el algoritmo SIR no tenga problemas con datos tan grandes
R = DeathsWorld + RecoveriesWorld
I = TotalCasesWorld - R
S = N - R - I

### Desde aproximadamente el 11 de marzo se presenta un comportamiento exponencial
### Calcular tasa de contagio y tasa de recuperación a partir de esa fecha
sI = I.loc['2020-03-11':'2020-04-05']
sR = R.loc['2020-03-11':'2020-04-05']
sS = S.loc['2020-03-11':'2020-04-05']


## Modelo SIR
def base_sir_model(init_vals, params, t):
    S_0, I_0, R_0 = init_vals
    S, I, R = [S_0], [I_0], [R_0]
    beta, gamma = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_S = S[-1] - (beta*S[-1]*I[-1])*dt
        next_I = I[-1] + (beta*S[-1]*I[-1] - gamma*I[-1])*dt
        next_R = R[-1] + (gamma*I[-1])*dt
        S.append(next_S)
        I.append(next_I)
        R.append(next_R)
    return S, I, R

# Parámetros
t_max = 100
dt = 1
t = np.linspace(0, t_max, int(t_max/dt) + 1)
N = 10000
init_vals = 1 - 1/N, 1/N, 0
beta, gamma = 0.23, 0.1    # Pico día 36, 15%
params = beta, gamma
ss, ii, rr = base_sir_model(init_vals, params, t)

i_temp = sI['Total'][0:25].values
i_temp = i_temp*(1/N)/i_temp[0]

# Interpolar datos para tener una grilla más fina
from scipy.interpolate import interp1d

f2 = interp1d(t[0:25], i_temp, kind='cubic')
tmax = 500
t_new = np.linspace(0,24,tmax)
i_temp_int = f2(t_new)

f2 = interp1d(t[0:25], ii[0:25], kind='cubic')
ii_int = f2(t_new)

# Plotting
import matplotlib
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 4}
matplotlib.rc('font', **font)

real_color = (0/255.,167/255.,255/255.)
simu_color = (203/255.,45/255.,111/255.)
whit_color = (240/255.,240/255.,240/255.)

count = 1
for i in range(tmax):
    count += 1
    # Plot inicial
    fig = plt.figure(facecolor='k',figsize=(1920/500, 1080/500), dpi=500)
    ax = fig.add_subplot(111, axisbelow=True)
    plt.axis([0, 24.5, 0, 0.00210])
    ax.plot(t_new[0:i+count],i_temp_int[0:i+count], color=(real_color), alpha=1, lw=0.5, label='real')
    ax.plot(t_new[0:i+count],ii_int[0:i+count], color=(simu_color), alpha=1, lw=0.5, label='modelo')
    ax.set_facecolor((0,0,0))

    # Leyenda
    legend = ax.legend(loc=(0.7,0.7),facecolor='k',edgecolor='k')
    for text in legend.get_texts():
        text.set_color(whit_color)

    # Axes (general)
    ax.set_xlabel('Días desde el inicio de la pandemia')
    ax.set_ylabel('Personas infectadas')

    # y axis
    ax.yaxis.set_tick_params(length=2.5)
    ax.yaxis.label.set_color(whit_color)
    ax.tick_params(axis='y', colors=whit_color)
    plt.yticks(np.arange(0, 0.0021, step=0.0004),['50.6K', '220.5K', '390.3K', '560.2K', '730.1K', '900K'])
    ax.yaxis.set_ticks_position('left')

    # x axis
    ax.xaxis.set_tick_params(length=2.5)
    ax.xaxis.label.set_color(whit_color)
    ax.tick_params(axis='x', colors=whit_color)
    plt.xticks(np.arange(0, 25, step=2))

    # Plot's border
    for spine in ('bottom', 'left'):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(whit_color)

    # Centrar en la imagen
    ancho = 0.8
    alto = 0.8
    x0 = 0.145
    y0 = 0.145
    ax.set_position([x0, y0, ancho, alto])

    # Guardar
    if i<=9:
        fname = 'img_00' + str(i) + '.png'
    if i>=10 & i<100:
        fname = 'img_0' + str(i) + '.png'
    if i>=100:
        fname = 'img_' + str(i) + '.png'
    print('Saving frame', fname)
    print(i)
    print(fname)
    plt.savefig(fname, facecolor='k')

# Generar video
# ffmpeg -r 30 -f image2 -s 1920x1080 -i img_%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p 24_500.mp4

