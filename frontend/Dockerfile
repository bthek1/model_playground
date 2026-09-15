# Production image for the SPA: build with Node, serve with nginx.
#
# The previous version of this file shipped the *Vite dev server* (NODE_ENV=
# development, `npm run dev`) and exposed 5173 while the app listens on 5180.
# See docker/nginx.conf for why serving the files is only half the job.

FROM node:20-slim AS build

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci

COPY . .

# Fails the build if a heavy dependency has leaked into the entry chunk.
RUN npm run build && npm run check:bundle


FROM nginx:1.27-alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/healthz || exit 1
