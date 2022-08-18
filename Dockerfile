# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
# Datascience notebook updated version on github: https://github.com/jupyter/docker-stacks/blob/master/datascience-notebook/Dockerfile
ARG BASE_CONTAINER=jupyter/scipy-notebook
FROM $BASE_CONTAINER 
#FROM --platform=linux/x86_64 $BASE_CONTAINER

LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

# Fix DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# # to install R non-interactively
# ENV DEBIAN_FRONTEND=noninteractive

# pre-requisites: no problem on arm
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    build-essential \
    fonts-dejavu \
    gfortran \
    gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Julia dependencies
# install Julia packages in /opt/julia instead of $HOME
ARG julia_version="1.7.3"
ENV JULIA_DEPOT_PATH=/opt/julia \
    JULIA_PKGDIR=/opt/julia \
    JULIA_VERSION="${julia_version}"

WORKDIR /tmp

# hadolint ignore=SC2046
# install platform specific julia.. for M1 mac, just use the x86 since the dockerfile is for x86
RUN set -x && \
    julia_arch=$(uname -m) && \
    julia_short_arch="${julia_arch}" && \
    echo "version is ${julia_arch}" && \
    if [ "${julia_short_arch}" == "x86_64" ]; then \
      julia_short_arch="x64"; \
    fi; \
    julia_installer="julia-${JULIA_VERSION}-linux-${julia_arch}.tar.gz" && \
    julia_major_minor=$(echo "${JULIA_VERSION}" | cut -d. -f 1,2) && \
    mkdir "/opt/julia-${JULIA_VERSION}" && \
    wget -q "https://julialang-s3.julialang.org/bin/linux/${julia_short_arch}/${julia_major_minor}/${julia_installer}" && \
    tar xzf "${julia_installer}" -C "/opt/julia-${JULIA_VERSION}" --strip-components=1 && \
    rm "${julia_installer}" && \
    ln -fs /opt/julia-*/bin/julia /usr/local/bin/julia

# Add Julia packages.
RUN julia -e 'import Pkg; Pkg.update()' && \
    julia -e 'import Pkg; Pkg.add(["PythonCall", "Distributions", "StaticArrays", "DifferentialEquations", "DataFrames", "DataFramesMeta", "BenchmarkTools", "PrettyTables" ])' && \
    julia -e 'using Pkg; pkg"precompile"'

# FAILS HERE
# install gnuplot 
# RUN apt-get -y install gnuplot 

WORKDIR $HOME