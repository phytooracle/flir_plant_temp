FROM ubuntu:18.04

WORKDIR /opt
COPY . /opt

USER root

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.6.15

RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update -y
RUN apt-get install -y wget \
                       gdal-bin \
                       libgdal-dev \
                       libspatialindex-dev \
                       build-essential \
                       software-properties-common \
                       apt-utils \
                       ffmpeg \
                       libsm6 \
                       libxext6 \
                       libtcmalloc-minimal4 \
                       libblosc-dev \
                       liblz4-dev \
                       libzstd-dev \
                       libsnappy-dev \
                       libbrotli-dev \
                        libjpeg-dev \
                        libopenjp2-7-dev \
                        libtiff5-dev \
                        zlib1g-dev \
                        libfreetype6-dev \
                        liblcms2-dev \
                        libwebp-dev \
                        tcl8.6-dev \
                        tk8.6-dev \
                        python-tk \
                        libharfbuzz-dev \
                        libfribidi-dev \
                        libxcb1-dev

RUN add-apt-repository ppa:ubuntugis/ubuntugis-unstable
RUN apt-get update
RUN apt-get install -y python3-pyproj
RUN apt-get install -y libgdal-dev
RUN apt-get install libffi-dev
RUN apt-get install -y libbz2-dev
RUN add-apt-repository ppa:ubuntugis/ppa
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
RUN export C_INCLUDE_PATH=/usr/include/gdal

# Download and extract Python sources
RUN cd /opt \
    && wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \                                              
    && tar xzf Python-${PYTHON_VERSION}.tgz

# Build Python and remove left-over sources
RUN cd /opt/Python-${PYTHON_VERSION} \ 
    && ./configure --enable-optimizations --with-ensurepip=install \
    && make install \
    && rm /opt/Python-${PYTHON_VERSION}.tgz /opt/Python-${PYTHON_VERSION} -rf

RUN apt-get update
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade wheel
RUN pip3 install cython
RUN pip3 install --upgrade cython
RUN pip3 install setuptools==57.5.0
RUN pip3 install GDAL==3.0.4
RUN pip3 install -r /opt/requirements.txt
RUN apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

ENTRYPOINT [ "/usr/bin/python3", "/opt/thermal_extraction.py" ]

