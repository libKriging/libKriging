FROM ubuntu:20.04
COPY install.sh /
RUN /install.sh && rm -rf /var/lib/apt/lists/*

# extra python usually installed using linux-macos/install.sh
RUN python3 -m pip install pytest numpy
RUN python3 -m pip install wheel # required ton compile pytest-valgrind
RUN python3 -m pip install pytest-valgrind
