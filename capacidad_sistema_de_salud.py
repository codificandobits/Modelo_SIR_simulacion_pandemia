import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

## Pre-procesar datos
ConfirmedCases_raw=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
Deaths_raw=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
Recoveries_raw=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

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

N = 10000       # Para convergencia de la simulación
R = DeathsWorld + RecoveriesWorld
I = TotalCasesWorld - R
S = N - R - I

### Desde aproximadamente el 11 de marzo se presenta un comportamiento exponencial
### Calcular tasa de contagio y tasa de recuperación a partir de esa fecha
sI = I.loc['2020-03-09':'2020-03-31']
sR = R.loc['2020-03-09':'2020-03-31']
sS = S.loc['2020-03-09':'2020-03-31']

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
t_max = 150
dt = 1
npts = 500
t = np.linspace(0, t_max, npts)
N = 10000
init_vals = 1 - 1/N, 1/N, 0
beta, gamma = 0.23, 0.1

# Simulación
params = beta, gamma
ss, ii, rr = base_sir_model(init_vals, params, t)

# Plotting
red = (255/255.,54/255.,60/255.)
white = (240/255.,240/255.,240/255.) 
blue = (0/255.,167/255.,255/255.)
light_blue = (38/255.,185/255.,209/255.)
green = (0/255.,176/255.,80/255.)
yellow = (255/255.,192/255.,0/255.)

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 4}
matplotlib.rc('font', **font)
count = 0
def plot_filled_frame(i,count,t,ii,rr):
    count += 1
    print(count)

    # Plot inicial
    fig = plt.figure(facecolor='k',figsize=(1920/500, 1080/500), dpi=500)
    ax = fig.add_subplot(111, axisbelow=True)
    plt.axis([0, 150.5, 0, 0.21])
    ax.plot(t[0:i+count],ii[0:i+count], color=(red), alpha=1, lw=0.5)
    ax.fill_between(t[0:i+count],ii[0:i+count], y2=0, color=(red), alpha=0.3)
    ax.plot(t, 0.01*np.ones(len(t)), color = (light_blue), lw=0.4, linestyle='--')
    ax.text(1,0.015,'Capacidad sistema de salud',color=light_blue)
    ax.set_facecolor((0,0,0))

    # Axes (general)
    ax.set_xlabel('Días desde el inicio de la pandemia')
    ax.set_ylabel('Porcentaje de la población mundial (%)')

    # y axis
    ax.yaxis.set_tick_params(length=2.5)
    ax.yaxis.label.set_color(white)
    ax.tick_params(axis='y', colors=white)
    plt.yticks(np.arange(0, 0.21, step=0.05),['', '5', '10', '15', '20'])
    ax.yaxis.set_ticks_position('left')

    # x axis
    ax.xaxis.set_tick_params(length=2.5)
    ax.xaxis.label.set_color(white)
    ax.tick_params(axis='x', colors=white)
    plt.xticks(np.arange(0, 150+1, step=20))

    # Plot's border
    for spine in ('bottom', 'left'):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(white)

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

for i in range(len(t)):
    plot_filled_frame(i,count,t,ii,rr)

# Create movie

ffmpeg -r 30 -f image2 -s 1920x1080 -i img_%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p 28_500.mp4



