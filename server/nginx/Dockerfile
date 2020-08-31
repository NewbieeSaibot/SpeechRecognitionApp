#Pull nginx image from docker hub
FROM nginx

# Remove nginx file and replace to a new one
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/