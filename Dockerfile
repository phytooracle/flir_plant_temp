FROM ubuntu:18.04

WORKDIR /opt
COPY . /opt

USER root

RUN apt-get update
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
                       libffi-dev

RUN wget https://www.python.org/ftp/python/3.8.5/Python-3.8.5.tgz
RUN tar -xzf Python-3.8.5.tgz
RUN cd Python-3.8.5/ && ./configure && make && make install
RUN apt-get install -y python3-pip

RUN add-apt-repository ppa:ubuntugis/ubuntugis-unstable
RUN apt-get update
RUN apt-get install -y libgdal-dev
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN wget http://download.osgeo.org/libspatialindex/spatialindex-src-1.7.1.tar.gz
RUN tar -xvf spatialindex-src-1.7.1.tar.gz
RUN cd spatialindex-src-1.7.1/ && ./configure && make && make install
RUN ldconfig
RUN add-apt-repository ppa:ubuntugis/ppa
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
RUN export C_INCLUDE_PATH=/usr/include/gdal
RUN apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

ENTRYPOINT [ "/usr/bin/python3", "/opt/thermal_extraction.py" ]