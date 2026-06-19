FROM mcr.microsoft.com/devcontainers/cpp:ubuntu-22.04

# This container cross-compiles the application for Raspberry Pi/aarch64.
RUN set -eux; \
    . /etc/os-release; \
    native_arch="$(dpkg --print-architecture)"; \
    if [ "$native_arch" != "arm64" ]; then \
      dpkg --add-architecture arm64; \
      for source_list in /etc/apt/sources.list /etc/apt/sources.list.d/*.list; do \
        [ -f "$source_list" ] || continue; \
        sed -i -E \
          -e "/^deb \[/ { /arch=/! s#^deb \[#deb [arch=${native_arch} #; }" \
          -e "/^deb [^[]/ s#^deb #deb [arch=${native_arch}] #" \
          "$source_list"; \
      done; \
      if ls /etc/apt/sources.list.d/*.sources >/dev/null 2>&1; then \
        for source_file in /etc/apt/sources.list.d/*.sources; do \
          awk -v native_arch="$native_arch" 'BEGIN { in_stanza = 0; saw_arch = 0 } \
               /^Types:/ { in_stanza = 1; saw_arch = 0 } \
               /^Architectures:/ { print "Architectures: " native_arch; saw_arch = 1; next } \
               /^$/ { if (in_stanza && !saw_arch) print "Architectures: " native_arch; print; in_stanza = 0; saw_arch = 0; next } \
               { print } \
               END { if (in_stanza && !saw_arch) print "Architectures: " native_arch }' "$source_file" > "$source_file.tmp"; \
          mv "$source_file.tmp" "$source_file"; \
        done; \
      fi; \
      printf '%s\n' \
        'Types: deb' \
        'URIs: http://ports.ubuntu.com/ubuntu-ports' \
        "Suites: ${VERSION_CODENAME} ${VERSION_CODENAME}-updates ${VERSION_CODENAME}-backports ${VERSION_CODENAME}-security" \
        'Components: main restricted universe multiverse' \
        'Architectures: arm64' \
        'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
        > /etc/apt/sources.list.d/ubuntu-arm64.sources; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake ninja-build \
    rsync openssh-client sshpass tmux \
    gdb-multiarch \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
    git curl wget unzip ca-certificates \
    python3 python3-pip python3-venv python3-dev \
    libopencv-dev \
 && rm -rf /var/lib/apt/lists/*

# Stage arm64 libjpeg into the aarch64 sysroot so the cross-compiler can link
# the RGB->JPEG encoder used by collect_data. (Headers jpeglib.h/jmorecfg.h/
# jerror.h live in /usr/include, the arch-specific jconfig.h in the multiarch
# include dir; the shared lib in the multiarch lib dir.)
RUN set -eux; \
    mkdir -p /tmp/jpeg-arm64/root /usr/aarch64-linux-gnu/include /usr/aarch64-linux-gnu/lib; \
    cd /tmp/jpeg-arm64; \
    apt-get update; \
    apt-get download libjpeg-turbo8:arm64 libjpeg-turbo8-dev:arm64; \
    for deb in *.deb; do dpkg-deb -x "$deb" /tmp/jpeg-arm64/root; done; \
    cp -a /tmp/jpeg-arm64/root/usr/include/*.h /usr/aarch64-linux-gnu/include/; \
    cp -a /tmp/jpeg-arm64/root/usr/include/aarch64-linux-gnu/*.h /usr/aarch64-linux-gnu/include/; \
    cp -a /tmp/jpeg-arm64/root/usr/lib/aarch64-linux-gnu/libjpeg.so* /usr/aarch64-linux-gnu/lib/; \
    rm -rf /tmp/jpeg-arm64 /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace
