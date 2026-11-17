FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Public landing page (frontend/landing, its own project), served at /welcome/
FROM node:22-alpine AS landing
WORKDIR /landing
COPY landing/package*.json ./
RUN npm ci
COPY landing/ ./
RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
COPY --from=landing /landing/dist /usr/share/nginx/html/welcome
EXPOSE 5173
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD wget -q -O /dev/null http://localhost:5173/ || exit 1
