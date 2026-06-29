# Build stage
FROM node:20-alpine as builder

WORKDIR /app

# Setup pnpm with proper environment
ENV PNPM_HOME="/root/.local/share/pnpm"
ENV PATH="${PNPM_HOME}:${PATH}"
RUN corepack enable && corepack prepare pnpm@latest --activate

COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

COPY . .
RUN pnpm run build

# Production stage
FROM node:20-alpine as production

WORKDIR /app

# Setup pnpm with environment
ENV PNPM_HOME="/root/.local/share/pnpm"
ENV PATH="${PNPM_HOME}:${PATH}"

RUN corepack enable && corepack prepare pnpm@latest --activate

# Install serve locally instead of globally
RUN pnpm add serve

COPY --from=builder /app/build ./build

EXPOSE 3000
CMD ["./node_modules/.bin/serve", "-s", "build", "-l", "3000"]
