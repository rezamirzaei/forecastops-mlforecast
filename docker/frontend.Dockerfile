FROM node:20-alpine AS build
WORKDIR /ui

COPY frontend/angular-ui/package*.json ./
RUN npm ci

COPY frontend/angular-ui/ ./
RUN npm run build

FROM nginx:1.27-alpine
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /ui/dist/angular-ui/browser /usr/share/nginx/html
EXPOSE 80
