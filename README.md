Costruir imágen que da soporte a la ejecución del desafio.

    docker build -t ds_ml .

Instanciar contenedor a partir de la imágen creada anteriormente.

    docker run -it --name ds_ml --rm -e TZ=America/Bogota -p 8888:8888 -v $PWD:/work ds_ml:latest