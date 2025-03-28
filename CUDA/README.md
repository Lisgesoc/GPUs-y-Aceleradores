HARDWARE LAB:
(Mirar el guion codigo device info)    

# Multiplicacion de matrices

    - V1: Ejecutar el producto de matrices en un unico bloque de hilos de un hilo

    - V2: Crear un grid de hilos de manera que los elemntos de la matriz C (matriz     resultado o producto) se calculen cada uno en un hilo  y que elemntos cercanos divididos en bloques de 16x16 se computen en un mismo bloque. Cada bloque a una SM sera de 16x16 hilos(elemntos)¡Pero esto tiene un problema!: Todos los accesos desde los threads se realizan sobre la memoria total de la GPU (Accesos notablemente lentos)
        TIEMPOS 1000 1000 1000:
            Tamaño en bytes: 4000000 
            Tiempo copia A:0.000407 
            Ancho de banda transf A: 9828.009766
            Tiempo copia B:0.000297 
            Ancho de banda transf B: 13468.013672
            Tiempo mul externo:0.001827 
            Rendimiento Kernel: 1094690.750000
            Tiempo copia C:0.000367 
            Ancho de banda transf C: 10899.182617

    - V3: Aprovechar los datos que trabajan una sm de manera que no necesite aceder cada hilo a los elementos de las matrices A y B. 
     Si la memoria interna de cada sm fuese infinita lo mas eficiente es que cada hilo de un bloque calculase un elem de C 
     que tomase la misma fila de A o columna de B. De esta manera solo tendira que cargarse los datos de una matriz una vez y los de la otra las veces necesarias
     ¡Pero esto tiene un problema!: El tamaño de k (elementos por fila de A o columna de B), es arbitrario y podria no caber en la mem compartida de una SM

    - V3.1: En esta version aprovecharemos la memoria compartida haciendo que los hilos que calcularan un bloque de la matriz producto hagan acesos a memoria sincronizados copiando uno a la vez todos los subloques necesarios para el calculo del bloque de C. Cuando se trae un subloque de A y B cada hilo multiplicara y sumara a una variable de resultaddos parciales la fila del subloque de A y la columna del subloque de B que corresponden a su elemento. Tras esto se syncrronizaran todos los hilos y copiaran los siguientes subloques necesarios.

# Transposicion de Matrices
    - V1: Cada hilo transpondera una fila/columna en memoria global lanzamos tantos hilos comoo filas/columnas haya
  
    - V2: Lanzamnos un hilo por elemento a transponder organizado en distintos bloques. Esto crea acesos a memoria alineados (Acceso a posiciones consecutivas en mem global) que reducen la latencia en el bus.

    - V3: Lanzamos un grid de hilos organizados en bq de dim=32, haran una carga paralela de elementos a mem compartida y luego cada hilo transpondera uno correspondiente. Esto reduce y mejora los accesos a memoria global, pero puede generar conflictos a la hora de acceder a la mem compartida (Son bancos de registros de 32 elementos cada banco, y el acceso consecutivo/simultaneo al mismo banco crea una latencia)

    - V4: Usamos la misma solucion que en la anterior version usando un padding(margen) al crear la variable de mem compartida de manera que los elementos estaran desplazados a partir del banco 0 pues su primera pos0=(y,x)=(1,1). Esto eliminara los conflictos a la hora de trasponer los elementos de la mem shared a la global accediendo desalineadamente.

# Apédice Streams
    - Un stream es una variable que nos permite delegar el proceso de coordinacion de hilos/kernels al controlador de nvidia. En ocasiones, la ejecucion de problemas simples como la incrementacion de todos los elementos de un vector o en general operaciones sin dependencias entre elementos, pueden no ser muy eficientes o por lo menos no la mejor solucion. Con los streams podemos dividir el trabajo del kernel original en varios subkernels que puedan ejecutarse en paralelo para resolver subproblemas. La ventaja se encuentra en la solapacion de accesos a memoria de un stream(carga de datos/kernel/descarga de datos) con la ejecucion de otros kernels 

# Line Assist
    -V0: 5000000ms aprox
    -V1: 80000ms aprox
