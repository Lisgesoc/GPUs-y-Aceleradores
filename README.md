# GPUs y Aceleradores
Repositorio remoto para la transferencia de codigo y archivos de la asignatura 

## Programacion paralela de GPUs NVIDIA en CUDA
    -Paralelizacion de computo en grid de hilos organizado en bloques
    -Optimizaciones de aceso a meemoria global:
        .Reduccion de acesos haciendo una copia en shared mem para uso de hilos en la misma SM
        .Copias colectivas a shared mem y syncronizacion antes del computo
    -Optimizacion de acesos a shared mem evitando acesos simultaneos o consecutivos al mismo banco de registros
    -Paralelizacion a nivel Kernel, creacion de streams dividiendo el kernel original (si es posible) en
     subkernels que resuelvan subproblemas no   dependientes y delegando la gestion de la ejecucion de los
     streams(Carga de datos / Kernel / Descarga de datos) al controlador de la grafica.