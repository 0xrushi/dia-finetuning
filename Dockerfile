FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git gcc g++ sudo \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && usermod -aG sudo $USERNAME \
    && mkdir -p /home/$USERNAME/app \
    && chown $USERNAME:$USERNAME /home/$USERNAME/app \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER $USERNAME
WORKDIR /home/$USERNAME/app

RUN python -m pip install --user --upgrade pip && pip install transformers datasets pydantic descript-audio-codec bitsandbytes


ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

WORKDIR /mnt

CMD ["bash"]