FROM nginx
COPY frontend /usr/share/nginx/html
COPY primeclue-api /usr/sbin
COPY run.sh /usr/sbin
CMD /bin/bash /usr/sbin/run.sh
