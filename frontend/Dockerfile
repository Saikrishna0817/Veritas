FROM node:24-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
# Public Turnstile site key (not a secret); empty = no human check on the forms. Vite reads it at build time.
ARG VITE_TURNSTILE_SITE_KEY=
ENV VITE_TURNSTILE_SITE_KEY=$VITE_TURNSTILE_SITE_KEY
RUN npm run build

# Public landing page (frontend/landing, its own project), served at /welcome/. It imports the code it shares
# with the app from ../shared, so the repository layout is kept (/src/landing next to /src/shared) and the shared
# files resolve their packages (three, gsap, react) from the landing page's own install.
FROM node:24-alpine AS landing
WORKDIR /src/landing
COPY landing/package*.json ./
RUN npm ci
COPY landing/ ./
COPY shared/ /src/shared/
RUN ln -s /src/landing/node_modules /src/node_modules && npm run build

FROM nginx:1.31-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
COPY --from=landing /src/landing/dist /usr/share/nginx/html/welcome
EXPOSE 5173
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD wget -q -O /dev/null http://localhost:5173/ || exit 1
