FROM ubuntu:20.04
ARG TARGETPLATFORM
ARG BUILDPLATFORM
RUN echo "Running on $BUILDPLATFORM, building for $TARGETPLATFORM"
COPY install.sh /
RUN /install-build.sh && rm -rf /var/lib/apt/lists/*
RUN /install-server.sh && rm -rf /var/lib/apt/lists/*
RUN apt update && apt install -y clang-tidy && rm -rf /var/lib/apt/lists/*

# extra python usually installed using linux-macos/install.sh
RUN python3 -m pip install pytest numpy
RUN python3 -m pip install wheel # required ton compile pytest-valgrind
RUN python3 -m pip install pytest-valgrind

