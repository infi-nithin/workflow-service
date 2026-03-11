FROM openresty/openresty:alpine
RUN apk add --no-cache bash curl
RUN rm -rf /usr/local/openresty/nginx/conf/nginx.conf
COPY nginx.conf /usr/local/openresty/nginx/conf/nginx.conf
COPY lua /lua
EXPOSE 8081
CMD ["/usr/local/openresty/bin/openresty", "-g", "daemon off;"]
