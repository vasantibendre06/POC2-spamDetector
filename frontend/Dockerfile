# Use Node.js to build the frontend
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package.json and yarn.lock first (for better caching)
COPY package.json yarn.lock ./

# Install dependencies
RUN yarn install

# Fix outdated caniuse-lite
RUN npx browserslist@latest --update-db

# Copy all project files
COPY . .

# Build the frontend
RUN yarn build

# Use Nginx as the web server
FROM nginx:alpine

# Remove the default Nginx config
RUN rm -rf /usr/share/nginx/html/*

# Copy build files from builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# Copy custom Nginx config (optional)
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port 80
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]

