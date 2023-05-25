# -*- coding: utf-8 -*-
"""
Código para analizar la frecuencia dominante/principal de vibración, mediante
la FFT rápida (fourier_freq(data, sampling_rate)) y la técnica del semiperiodo
(crossings_freq(data, sampling_rate)), a partir de los datos exportados del 
sismógrafo como fichero de texto.

Es imperativo analizar gráficamente la señal. 
"""

import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Ejemplos introductorios: comentar para introducir datos tras los ejemplos

# La frecuenca dominante hace referencia a aquella de mayor energía dentro
# de todo el espectro de frecuancias. La principal, en cambio, es aquella 
# correspondiente a la velocidad pico de partícula (ppv).

# sampling rate y sampling interval
sampling_rate = 2048 # number of measurements per second
ts = 1.0/sampling_rate

# ejemplo 1_Onda simple amortiguada...............................
time = np.arange(0, 4/10, ts) # duración de la onda
freq = 10. # frequency (Hertz)
data = (0.75*np.sin(2*np.pi*freq*time))*np.exp(-8*time)
data=np.insert(data, 0,0)
#................................................................
# ejemplo 2_Onda con componentes de diferentes frecuencias con entrada en 
# distintos tiempos. Decomentar para examinarlo

# time = np.arange(0, 2/10, ts) # duración de la primera onda
# freq = 10. # frequency (Hertz)
# # primera componente 
# data1 = 0.75 * np.sin(2*np.pi*freq*time) * np.exp(-8*time)

# # segunda componente
# time = np.arange(2/10, 2/10+6/30, ts)
# freq = 30.
# data2 = 0.5* np.sin(2*np.pi*freq*time) *np.exp(-8*time)

# # se unen los datos
# data =  np.concatenate((data1,data2))
# time = np.arange(0, 2/10+6/30, ts)

#..............................................................................
# # gráfica: decomentar para mostrar las ondas de los ejemplos
# plt.figure(0)
# plt.figure(figsize = (8, 6))
# plt.plot(time, data, 'r')
# plt.xlabel('time')
# plt.ylabel('Amplitude')

# plt.show()
#...........................................................................
#############################################################################♥
# # decomentar para introducir datos del fichero de texto
# import pandas as pd
# # lectura de datos exportados del sismógrafo ('data.csv': nombre del fichero)
# df_x = pd.read_csv('data.csv', sep=",") 
# time = df_x['Time']

# # se escoge el Channel que se quiere analizar (1-6)
# canal = 1
# sampling_rate = 2048 # frecuencia de muestero: número de medidas por segundo
# data = df_x['Channel: ' + str(canal)]
##############################################################################

def fourier_freq(data, sampling_rate):
    ''' Obtiene la frecuencia principal aplicando la FFT, y grafica los 
    resultados
    Puede que sea conveniente aplicar algún filtro previo a la señal.'''
    
    # velocidad pico y escala de tiempos
    ppvmax = np.max(abs(data))
    time = np.arange(len(data))/sampling_rate
    
    # transformada de la señal: se pasa a compleja, y frecuencias de trabajo
    fft_data = np.fft.fft(data)
    n_data = len(fft_data)
    freqs = np.fft.fftfreq(n_data)*sampling_rate
    
    # se calculan los modulos de la transformada
    magnitud = np.abs(fft_data)
    
    # se ordenan de mayor a menor valor y se obtienen las frecuencias asociadas
    # primero se obtienen los índices de menor valor a mayor valor
    sort_magnitud_index = np.argsort(magnitud)
    # se invierten para tener los índices de mayor a menor magnitud
    sort_magnitud_index = sort_magnitud_index[::-1]
    # y se ordenan las magnitudes de mayor a menor
    magnitud_sort = magnitud[sort_magnitud_index]

    # se ordenan las frecuencias según el orden de las magnitudes y se filtran
    freq_sort = (freqs[sort_magnitud_index])
    freq_sort = freq_sort[freq_sort>=0]
    
    # se obtiene el valor de la frecuencia dominante (debe ser la primera)
    dominant_freq = freq_sort[0]

    # grafico la onda: siempre realizar análisis gráfico
    plt.figure(10)
    plt.plot(1000*time, data, linestyle='solid', label='wave')
    plt.xlabel('time*1000')
    plt.ylabel('velocity')
    plt.margins(0.05)
    
    # grafico el espectro: siempre realizar análisis gráfico
    fig, ax = plt.subplots()
    ax.stem(freqs, magnitud, use_line_collection=True)
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('Spectrum Magnitude')
    ax.set_xlim(-sampling_rate / 2, sampling_rate / 2) # nyquist frequency
    
    # decomentar para ajustar el rango de visualización de frecuencias
    ax.set_xlim(0, 180)
    
    return (dominant_freq, ppvmax, freq_sort, magnitud_sort)

resultado_fourier = fourier_freq(data, sampling_rate)

def crossings_freq(data, sampling_rate):
    """ Obtiene la frecuencia principal mediante la técnica del semiperiodo,
    buscando los cruces de la señal (data) con el valor cero.
    Puede que sea conveniente aplicar algún filtro previo a la señal."""
    
    # se eliminan, de los datos, la cadena inicial de ceros y se introduce un
    # valor pequeño en data[0] para marcar el inicio de la onda
    
    if data[0] == 0:
        data = np.trim_zeros(data)
        if data[0]>0: data = np.insert(data,0,-0.00000001)
        elif data[0]<0: data = np.insert(data,0,0.000000001)
    
    # velocidad pico y su posición
    ppv_max = np.max(abs(data))
    ppv_ind =np.argmax(abs(data))
    
    # índices de los puntos en donde hay un cambio de signo (cruces con cero)
    index = np.where(np.diff(np.signbit(data)))[0]
    
    # se obtienen las ppv entre los cruces con zero
    ppv = np.zeros(shape=index.shape)
    for i in range(len(index)-1):
        start = index[i]
        end = index[i+1]
        ppv[i] = np.max(abs(data[start:end]))
        
    # interpolación para ajustar, posteriormente, los tiempos de cruce
    crossings = [i - data[i] / (data[i+1] - data[i]) for i in index]
    axi_cross = np.zeros_like(crossings)
    
    # tiempos en los que ocurre el cruce y escala de tiempos
    t_crossing = np.array(crossings)/sampling_rate
    time = np.transpose(np.array([[i for i in range(len(data))]])/sampling_rate)
    
    # salida gráfica de los cruces y ppv (ppv dibujado a la izda de su máximo)
    plt.figure(20)
    plt.plot(t_crossing*1000, axi_cross, marker='+', linestyle='none', \
             label='crossing')
    plt.plot(t_crossing*1000, ppv, marker='x', linestyle='none', label='abs(ppv)')
    # grafico la onda: siempre realizar análisis gráfico
    plt.plot(time*1000, data, linestyle='solid', label='wave')
    plt.xlabel('time*1000')
    plt.ylabel('velocity')
    plt.legend()
    plt.margins(0.05)
   
    # índice correspondiente a la frecuencia de ppv (el menor valor de dist>0)
    dist = crossings - ppv_ind
    pos = np.array(np.where(dist>0))[0][0]
    
    # cálculo de las frecuencias a partir de los puntos de cruce
    freq = sampling_rate/(np.diff(crossings) * 2)
    principal_freq = freq[pos-1]
    
    # se grafica
    fig, ax = plt.subplots()
    ax.plot(freq, ppv[:-1], marker='x', linestyle='none')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('ppv')
    ax.set_xlim(0, 180)
    
    return principal_freq, ppv_max, freq, ppv

resultado_zerocrossing = crossings_freq(data, sampling_rate)
