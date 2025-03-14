# Use a Node.js image for frontend
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Copy frontend files
COPY frontend /app

# Install dependencies
RUN npm install

# Build the frontend
RUN npm run build

# Use an Nginx server to serve the React app
FROM nginx:alpine
COPY --from=0 /app/build /usr/share/nginx/html

# Expose Nginx default port
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
