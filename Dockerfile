FROM ubuntu:18.04

WORKDIR /opt
COPY . /opt

USER root

RUN apt-get update
RUN apt-get install -y python3.6-dev \
                       python3-pip \
                       wget \
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

# RUN export export LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4"
RUN add-apt-repository ppa:ubuntugis/ubuntugis-unstable
RUN apt-get update
RUN apt-get install -y libgdal-dev
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install --upgrade wheel
RUN pip3 install cython
RUN pip3 install --upgrade cython
# RUN pip3 install setuptools==57.5.0
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

