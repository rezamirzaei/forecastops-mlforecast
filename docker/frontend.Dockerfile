# syntax=docker/dockerfile:1

FROM node:20-alpine AS build
WORKDIR /ui

# Install dependencies first (layer cache)
COPY frontend/angular-ui/package*.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci

# Copy source and build
COPY frontend/angular-ui/ ./
RUN npm run build

FROM nginx:1.27-alpine
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /ui/dist/angular-ui/browser /usr/share/nginx/html
EXPOSE 80
