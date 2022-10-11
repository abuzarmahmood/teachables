# Download docker image
docker pull abuzarmahmood/pymc3:working
# Run docker image with:
# 1) Code directory mounted at `/app`
# 2) jupyter notebook command
# 3) Running in interacitve mode
docker run -v ~/Downloads/teachables:/app -it -p 8888:8888 --network=host abuzarmahmood/pymc3:working jupyter notebook --allow-root
