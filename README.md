# Description

## Overview

Python package intended to manipulate, analyze and visualize geospatial data. It was developed in the context of my [PhD thesis](https://github.com/SandroAlex/phd) in the field of atmospheric physics. The PhD thesis text (in Portuguese) is publicly available at [Biblioteca Digital de Teses e Dissertações da USP](https://www.teses.usp.br/teses/disponiveis/43/43134/tde-29092023-230453/pt-br.php).

## Containerization

Using the [Dockerfile](./Dockerfile) at the root directory, you can test cdlearn package in an isolated environment. First you can build the image with the following command:
```shell
docker build --tag cdlearn-user/cdlearn:latest --file ./Dockerfile .
```
then you can run the resulting container using this command:
```shell
docker run --rm --interactive --detach --entrypoint /bin/bash --name cdlearn-container cdlearn-user/cdlearn:latest
```
After the container is set to run, enter it in interactive mode:
```shell
docker exec --interactive --tty cdlearn-container /bin/bash
```
Finally, after exiting the container, stop it:
```shell
docker stop cdlearn-container
```
Alternatively, if you have Visual Studio Code installed on your machine, we have configured a [devcontainer](./devcontainer) full-featured environment where it is possible to develop and use cdlearn package. See more information at [Developing inside a Container](https://code.visualstudio.com/docs/devcontainers/containers).