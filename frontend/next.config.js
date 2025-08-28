/** @type {import('next').NextConfig} */
const nextConfig = {
  // Build configuration
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: false,
  },
  
  // Performance optimizations
  swcMinify: true,
  compress: true,
  
  // Output configuration for Vercel
  output: 'standalone',
  
  // Experimental features
  experimental: {
    typedRoutes: true,
    optimizeCss: true,
    serverComponentsExternalPackages: ['sharp'],
    webVitalsAttribution: ['CLS', 'LCP'],
  },
  
  // Image optimization
  images: {
    formats: ['image/webp', 'image/avif'],
    minimumCacheTTL: 86400, // 24 hours
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**',
      },
      // Add specific patterns for better security in production
      {
        protocol: 'https',
        hostname: 'images.unsplash.com',
      },
      {
        protocol: 'https',
        hostname: 'avatars.githubusercontent.com',
      },
    ],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
  },
  
  // Security headers
  poweredByHeader: false,
  
  // API route rewrites
  async rewrites() {
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 
                      process.env.NEXT_PUBLIC_API_URL || 
                      'http://localhost:8001';
    
    return [
      // Proxy API requests to backend
      {
        source: '/api/backend/:path*',
        destination: `${backendUrl}/:path*`,
      },
      // Direct API v1 proxy
      {
        source: '/api/v1/:path*',
        destination: `${backendUrl}/api/v1/:path*`,
      },
      // Health checks
      {
        source: '/api/health',
        destination: `${backendUrl}/health`,
      },
      // Documentation
      {
        source: '/api/docs',
        destination: `${backendUrl}/docs`,
      },
      {
        source: '/api/redoc',
        destination: `${backendUrl}/redoc`,
      },
    ];
  },
  
  // Redirect configuration
  async redirects() {
    return [
      // Redirect old paths
      {
        source: '/home',
        destination: '/',
        permanent: true,
      },
      // Redirect API documentation to external docs if needed
      {
        source: '/docs',
        destination: '/api/docs',
        permanent: false,
      },
    ];
  },
  
  // Custom headers for security and performance
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          // Security headers
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on'
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff'
          },
          {
            key: 'X-Frame-Options',
            value: 'DENY'
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block'
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin'
          },
          // Performance headers
          {
            key: 'X-UA-Compatible',
            value: 'IE=edge'
          },
        ],
      },
      {
        // Cache static assets aggressively
        source: '/_next/static/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
      {
        // Cache images for 1 day
        source: '/images/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=86400',
          },
        ],
      },
    ];
  },
  
  // Webpack customizations for optimization
  webpack: (config, { dev, isServer }) => {
    // Production optimizations
    if (!dev) {
      // Minimize bundle size
      config.optimization.splitChunks = {
        chunks: 'all',
        cacheGroups: {
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name: 'vendors',
            chunks: 'all',
          },
        },
      };
    }
    
    // Add source maps in development
    if (dev) {
      config.devtool = 'eval-source-map';
    }
    
    return config;
  },
  
  // Environment variables validation
  env: {
    CUSTOM_KEY: process.env.CUSTOM_KEY,
  },
  
  // Serverless function configuration
  serverRuntimeConfig: {
    // Will only be available on the server side
    mySecret: process.env.MY_SECRET,
  },
  publicRuntimeConfig: {
    // Will be available on both server and client
    staticFolder: '/static',
  },
};

module.exports = nextConfig;