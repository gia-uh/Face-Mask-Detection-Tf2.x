FROM autogoal/autogoal

USER root 
RUN apt update && apt install -y libgl1-mesa-glx
RUN pip install opencv-python fdet
RUN chown coder:coder /opt/dev/cache/

USER coder
COPY . /home/coder/facemask
WORKDIR /home/coder/facemask
