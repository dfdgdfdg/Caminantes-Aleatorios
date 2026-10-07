#Simulo el camino aleatorio
import numpy as np
import pandas as pd
import geopandas as gpd

def simular_caminata(
    G,
    clases,
    nodo_inicial,
    max_pasos,
    rng,
    guardar_recorrido = True
):
    """
    Simula una caminata aleatoria desde nodo_inicial. Retorna:
    - tiempos_encuentro[k]: primer tiempo en que se observaron k clases
    - cantidad_clases_t[t]: cantidadlases observadas hasta el tiempo t
    """
    #Cantidad de clases
    cantidad_total_clases = len(set(clases.values()))

    #Nodo donde arranca el caminante
    nodo_actual = nodo_inicial

    #Empiezo con el recorrido, el primer nodo del recorrido es el inicial
    if guardar_recorrido:
        recorrido = [(nodo_inicial, clases[nodo_inicial])]
    else:
        recorrido = None
        #No guardo el recorrido
    
    #Clase del nodo donde arranca el caminante
    clases_observadas = {
        clases[nodo_actual]
    }

    #Empieza con todos 1s, va reemplazando en el elemento "t" segun la cantidad de clases encontradas en el paso t.
    cantidad_clases_t = np.ones(
        max_pasos + 1,
        dtype = np.int8
    )

    #Cuando se encuentra el caminante con X cantidad de clases?
    tiempos_encuentro = {
        k: None
        for k in range(1,cantidad_total_clases + 1)
    }

    #En t=0 ya conoce 1 clase, la propia:
    tiempos_encuentro[1] = 0

    #Representa la iteracion, los pasos de la caminata que va haciendo la persona
    for tiempo in range(1, max_pasos + 1):

        #vecinos del nodo actual
        vecinos = list(G.neighbors(nodo_actual))

        #Elijo aleatoriamente y equiprobablemente un vecino dentro de la lista de vecinos del nodo actual.
        #Uso "len(vecinos)" porque capaz estoy parado en una celda que tiene 3 vecinos, o en una esquina, con 2.
        indice_elegido = rng.integers(
            low=0,
            high=len(vecinos)
        )

        #Me muevo al nuevo nodo
        nodo_actual = vecinos[indice_elegido]

        #Agrego el nuevo nodo a mi recorrido
        if guardar_recorrido:
            recorrido.append((nodo_actual, clases[nodo_actual]))

        cantidad_anterior = len(clases_observadas)

        #Agrego la clase del nuevo nodo a mi lista de clases observadas
        clases_observadas.add(
            clases[nodo_actual]
        )

        #Actualizo cantidad de clases
        cantidad_actual = len(clases_observadas)

        #Actualizo el array de cantidad de clases visitadas en cada paso
        cantidad_clases_t[tiempo] = cantidad_actual

        #Si es la primera vez que descubro una nueva clase, en tiempos_encuentro actualizo el tiempo en encontrar X cantidad de clases.
        if cantidad_actual > cantidad_anterior and tiempos_encuentro[cantidad_actual] is None:
            tiempos_encuentro[cantidad_actual] = tiempo
            # Si es la primera vez que descubro esta clase, actualizo el tiempo
        
        #Ya miré todas las clases, corto la caminata.
        if cantidad_actual == cantidad_total_clases:
            cantidad_clases_t[tiempo:] = cantidad_total_clases
            break

    return tiempos_encuentro, cantidad_clases_t, recorrido


def estimar_tiempos_nodo(
    G,
    clases,
    nodo_inicial,
    numero_simulaciones=100,
    max_pasos=500,
    semilla=123,
    nombre_configuracion = None
):
    """
    Ejecuta muchas caminatas desde un mismo nodo inicial. Calcula estadísticas o metricas como medianas y proporciones de caminatas que no alcanzaron cada objetivo.
    """

    cantidad_total_clases = len(set(clases.values()))

    #Validacion nodo inicial
    if nodo_inicial not in G:
        raise ValueError(
            f"El nodo inicial {nodo_inicial} no pertenece al grafo."
        )

    # Validación nodos sin clase
    nodos_sin_clase = [
        nodo
        for nodo in G.nodes
        if nodo not in clases
    ]
    if len(nodos_sin_clase) > 0:
        raise ValueError(
            f"Hay {len(nodos_sin_clase)} nodos sin clase asignada."
        )

    # Generador aleatorio específico para este nodo
    rng = np.random.default_rng(seed=semilla)

    # Para cada k guardo un tiempo X_k por simulación
    valores_xk = {
        k: []
        for k in range(2,cantidad_total_clases + 1)
    }

    # Repito muchas caminatas desde el mismo origen
    for simulacion in range(numero_simulaciones):

        tiempos, cantidad_clases_t, _ = simular_caminata(
            G=G,
            clases=clases,
            nodo_inicial=nodo_inicial,
            max_pasos=max_pasos,
            rng=rng,
            guardar_recorrido=False
        )

        # Guardar X_k para cada nivel de diversidad
        for k in range(2,cantidad_total_clases + 1):
            if tiempos[k] is None:
                valores_xk[k].append(np.nan)
            else:
                valores_xk[k].append(tiempos[k])

    # lo paso a array
    for k in valores_xk:
        valores_xk[k] = np.asarray(valores_xk[k], dtype=float)

    # Información general del nodo
    resultado = {
    "configuracion":nombre_configuracion,
    "nodo": nodo_inicial,
    "clase_inicial": clases[nodo_inicial],
    "simulaciones": numero_simulaciones,
    "max_pasos": max_pasos,
    "cantidad_total_clases": cantidad_total_clases
    }

    # Estadísticas para cada k
    for k, valores in valores_xk.items():

        alcanzados = valores[~np.isnan(valores)] 
        proporcion_no_alcanzo = np.isnan(valores).mean()

        if len(alcanzados) > 0:
            tiempo_promedio = np.nanmean(valores)
            mediana = np.nanmedian(valores)
            desvio = np.nanstd(valores, ddof=1)
            minimo = np.nanmin(valores)
            maximo = np.nanmax(valores)
        else:
            tiempo_promedio = np.nan
            mediana = np.nan
            desvio = np.nan
            minimo = np.nan
            maximo = np.nan

        # Columnas dinámicas:
        resultado[f"t{k}"] = tiempo_promedio
        resultado[f"mediana_x{k}"] = mediana
        resultado[f"desvio_x{k}"] = desvio
        resultado[f"minimo_x{k}"] = minimo
        resultado[f"maximo_x{k}"] = maximo
        resultado[f"no_alcanzo_{k}"] = proporcion_no_alcanzo

    return resultado, valores_xk

def estimar_tiempos_nodo_completo(
    G,
    clases,
    nodo_inicial,
    numero_simulaciones=100,
    max_pasos=500,
    semilla=123,
    nombre_configuracion=None
):

    cantidad_total_clases = len(set(clases.values()))

    if nodo_inicial not in G:
        raise ValueError(f"El nodo inicial {nodo_inicial} no pertenece al grafo.")

    nodos_sin_clase = [nodo for nodo in G.nodes if nodo not in clases]

    if len(nodos_sin_clase) > 0:
        raise ValueError(f"Hay {len(nodos_sin_clase)} nodos sin clase asignada.")

    rng = np.random.default_rng(seed=semilla)
    valores_xk = {k: [] for k in range(2,cantidad_total_clases + 1)}
    resultados_caminatas = []

    for simulacion in range(numero_simulaciones):

        tiempos, cantidad_clases_t, _ = simular_caminata(
            G=G,
            clases=clases,
            nodo_inicial=nodo_inicial,
            max_pasos=max_pasos,
            rng=rng,
            guardar_recorrido=False
        )

        # Información general de esta caminata
        fila_caminata = {
            "configuracion": nombre_configuracion,
            "nodo": nodo_inicial,
            "clase_inicial": clases[nodo_inicial],
            "simulacion": simulacion,
            "semilla_nodo": semilla,
            "max_pasos": max_pasos
        }

        # Guardar cada X_k de esta caminata
        for k in range(2, cantidad_total_clases + 1):
            if tiempos[k] is None:
                valor_xk = np.nan
            else:
                valor_xk = float(tiempos[k])

            valores_xk[k].append(valor_xk)
            fila_caminata[f"x{k}"] = valor_xk
            fila_caminata[f"alcanzo_{k}"] = not np.isnan(valor_xk)

        resultados_caminatas.append(fila_caminata)

    for k in valores_xk:
        valores_xk[k] = np.asarray(valores_xk[k],dtype=float)

    resultado = {
        "configuracion": nombre_configuracion,
        "nodo": nodo_inicial,
        "clase_inicial": clases[nodo_inicial],
        "simulaciones": numero_simulaciones,
        "max_pasos": max_pasos,
        "cantidad_total_clases": cantidad_total_clases
    }

    for k, valores in valores_xk.items():

        alcanzados = valores[~np.isnan(valores)]
        proporcion_no_alcanzo = (np.isnan(valores).mean())
        cantidad_alcanzo = len(alcanzados)
        cantidad_no_alcanzo = (numero_simulaciones - cantidad_alcanzo)

        if cantidad_alcanzo > 0:
            tiempo_promedio = np.mean(alcanzados)
            mediana = np.median(alcanzados)
            desvio = np.std(alcanzados,ddof=1)
            minimo = np.min(alcanzados)
            maximo = np.max(alcanzados)
        else:
            tiempo_promedio = np.nan
            mediana = np.nan
            desvio = np.nan
            minimo = np.nan
            maximo = np.nan

        # Estadísticas dinámicas
        resultado[f"t{k}"] = (tiempo_promedio)
        resultado[f"mediana_x{k}"] = (mediana)
        resultado[f"desvio_x{k}"] = (desvio)
        resultado[f"minimo_x{k}"] = (minimo)
        resultado[f"maximo_x{k}"] = (maximo)
        resultado[f"cantidad_alcanzo_{k}"] = (cantidad_alcanzo)
        resultado[f"cantidad_no_alcanzo_{k}"] = (cantidad_no_alcanzo)
        resultado[f"no_alcanzo_{k}"] = (proporcion_no_alcanzo)

    resultados_caminatas = pd.DataFrame(resultados_caminatas)
    return resultado, valores_xk, resultados_caminatas

radios = gpd.read_file("radios-censales-2022.geojson")

columnas_codigo = ["codigo","cod_prov","cod_dep","fraccion","radio"]
for columna in columnas_codigo: radios[columna] = (radios[columna].astype("string").str.strip()) #Todo a string y sin espacios
radios["codigo"] = radios["codigo"].str.zfill(9)
radios["cod_prov"] = radios["cod_prov"].str.zfill(2)
radios["cod_dep"] = radios["cod_dep"].str.zfill(3)
radios["fraccion"] = radios["fraccion"].str.zfill(2)
radios["radio"] = radios["radio"].str.zfill(2)
radios["codigo_reconstruido"] = ( radios["cod_prov"] + radios["cod_dep"] + radios["fraccion"] + radios["radio"])

PARTIDOS_SELECCIONADOS ={
 "035":"Avellaneda",
 "371":"General San Martín",
 "427":"La Matanza",
 "434":"Lanús",
 "490":"Lomas de Zamora",
 "840":"Tres de Febrero",
 "861":"Vicente López",
 "756":"San Isidro",
 "568":"Morón"
}

condicion_caba = (radios["cod_prov"] == "02") #CABA
condicion_partidos = (radios["cod_prov"] == "06") & (radios["cod_dep"].isin(PARTIDOS_SELECCIONADOS.keys())) #AMBA, los partidos mencionados arriba
exclusion_la_matanza_periferico = (radios["cod_dep"] == "427") & (radios["fraccion"].astype(int) > 80 ) #Saco de La Matanza partidos como Virrey del Pino, Gonzalez Catan y 20 de Junio porque están muy lejos de CABA.

radios_estudio = radios.loc[condicion_caba | (condicion_partidos & (~exclusion_la_matanza_periferico)) ].copy()
radios_estudio = radios_estudio.reset_index(drop=True)
radios_estudio["prov_dep"] = radios_estudio["cod_prov"] + radios_estudio["cod_dep"]
radios_estudio["jurisdiccion"] = np.where(radios_estudio["cod_prov"] == "02","CABA","Provincia de Buenos Aires")
radios_estudio["area_nombre"] = np.where(radios_estudio["cod_prov"] == "02","CABA",radios_estudio["cod_dep"].map(PARTIDOS_SELECCIONADOS))
