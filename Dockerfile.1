##Dockerfile
#https://hub.docker.com/r/ducatel/visual-studio-linux-build-box/

# FROM dmccloskey/docker-openms-contrib:develop
FROM dmccloskey/docker-openms-contrib:smartPeak
# FROM dmccloskey/docker-openms-contrib:smartpeak2

USER root
COPY docker-entrypoint.sh /usr/local/bin/

## use with dmccloskey/docker-openms-contrib:develop
# RUN apt-get update && \
# 	DEBIAN_FRONTEND=noninteractive && \
# 	apt-get install -y \
# 	gawk \
# 	openssh-server \
# 	gdb \
# 	gdbserver \
# 	sudo \
# 	build-essential \
# 	git && \
# 	apt-get clean && \
#     rm -r /var/lib/apt/lists/* && \
## use with dmccloskey/docker-openms-contrib:smartPeak or smartpeak2
RUN DEBIAN_FRONTEND=noninteractive && \
	apk add --no-cache \
	valgrind && \
	##change permissions
	ln -s /usr/local/bin/docker-entrypoint.sh / && \
	chmod +x /usr/local/bin/docker-entrypoint.sh && \
	mkdir /home/user/code

# USER user
CMD ["docker-entrypoint.sh"]

#docker run -it --name=cpp_openms_1 -v //C/Users/domccl/GitHub/smartPeak_cpp/:/home/user/code/ --privileged --security-opt seccomp:unconfined dmccloskey/docker-openms-vscode /bin/bash