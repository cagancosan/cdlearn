# Description

## Overview

Python package intended to manipulate, analyze and visualize geospatial data. It was developed in the context of my [PhD thesis](https://github.com/SandroAlex/phd) in the field of atmospheric physics. The PhD thesis text (in Portuguese) is publicly available at [Biblioteca Digital de Teses e Dissertações da USP](https://www.teses.usp.br/teses/disponiveis/43/43134/tde-29092023-230453/pt-br.php).

## Docker Local Commands

```shell
docker build \
    --tag cdlearn-user/cdlearn:latest \
    --file ./Dockerfile .
```

```shell
docker run \
    --rm \
    --interactive \
    --detach \
    --entrypoint /bin/bash \
    --name cdlearn-container \
    cdlearn-user/cdlearn:latest
```

```shell
docker exec \
    --interactive \
    --tty \
    cdlearn-container \
    /bin/bash
```

```shell
docker stop cdlearn-container
```