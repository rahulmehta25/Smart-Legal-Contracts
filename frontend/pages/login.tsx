import React, { useState } from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';
import { Eye, EyeOff, Shield, Mail, Lock, ArrowRight, Github, Google } from 'lucide-react';
import { toast } from 'react-hot-toast';
import { useAuth } from '../src/contexts/AuthContext';
import { useTheme } from '../src/contexts/ThemeContext';

export default function LoginPage() {
  const router = useRouter();
  const { login } = useAuth();
  const { isDark } = useTheme();
  
  const [email, setEmail] = useState('demo@example.com');
  const [password, setPassword] = useState('password');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!email || !password) {
      toast.error('Please fill in all fields');
      return;
    }

    setIsLoading(true);
    
    try {
      await login(email, password);
      toast.success('Successfully logged in!');
      
      // Redirect to dashboard or intended page
      const returnUrl = router.query.returnUrl as string || '/dashboard';
      router.push(returnUrl);
    } catch (error) {
      toast.error('Invalid email or password');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSocialLogin = async (provider: 'google' | 'github') => {
    try {
      // Mock social login - replace with actual implementation
      toast.loading(`Connecting to ${provider}...`);
      await new Promise(resolve => setTimeout(resolve, 2000));
      toast.success(`Successfully connected to ${provider}!`);
      router.push('/dashboard');
    } catch (error) {
      toast.error(`Failed to connect to ${provider}`);
    }
  };

  return (
    <div id="login-container" className={`min-h-screen flex ${isDark ? 'bg-gray-900' : 'bg-gray-50'}`}>
      {/* Left Side - Branding */}
      <div id="login-branding" className={`hidden lg:flex lg:w-1/2 ${isDark ? 'bg-gray-800' : 'bg-primary-600'} flex-col justify-center px-12`}>
        <div id="branding-content" className="max-w-md">
          <div id="logo-section" className="flex items-center space-x-3 mb-8">
            <div id="logo-icon" className="p-3 bg-white bg-opacity-20 rounded-lg">
              <Shield className="h-8 w-8 text-white" />
            </div>
            <h1 id="brand-name" className="text-3xl font-bold text-white">
              ArbiScan Pro
            </h1>
          </div>
          
          <h2 id="welcome-title" className="text-4xl font-bold text-white mb-6">
            Welcome back
          </h2>
          
          <p id="welcome-description" className="text-xl text-primary-100 mb-8 leading-relaxed">
            Continue analyzing legal documents with our AI-powered arbitration clause detection system.
          </p>

          {/* Features List */}
          <div id="features-list" className="space-y-4">
            <div id="feature-1" className="flex items-center space-x-3">
              <div className="w-2 h-2 bg-white rounded-full"></div>
              <span className="text-primary-100">Advanced AI Analysis</span>
            </div>
            <div id="feature-2" className="flex items-center space-x-3">
              <div className="w-2 h-2 bg-white rounded-full"></div>
              <span className="text-primary-100">Real-time Document Processing</span>
            </div>
            <div id="feature-3" className="flex items-center space-x-3">
              <div className="w-2 h-2 bg-white rounded-full"></div>
              <span className="text-primary-100">Comprehensive Reporting</span>
            </div>
            <div id="feature-4" className="flex items-center space-x-3">
              <div className="w-2 h-2 bg-white rounded-full"></div>
              <span className="text-primary-100">Enterprise-grade Security</span>
            </div>
          </div>
        </div>
      </div>

      {/* Right Side - Login Form */}
      <div id="login-form-section" className="flex-1 flex flex-col justify-center px-6 sm:px-12 lg:px-16">
        <div id="form-container" className="w-full max-w-md mx-auto">
          {/* Mobile Logo */}
          <div id="mobile-logo" className="lg:hidden text-center mb-8">
            <div id="mobile-logo-icon" className="inline-flex items-center justify-center w-16 h-16 bg-primary-600 rounded-xl mb-4">
              <Shield className="h-8 w-8 text-white" />
            </div>
            <h1 id="mobile-brand-name" className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
              ArbiScan Pro
            </h1>
          </div>

          {/* Form Header */}
          <div id="form-header" className="text-center mb-8">
            <h2 id="form-title" className={`text-3xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
              Sign in to your account
            </h2>
            <p id="form-description" className={`mt-2 text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              Or{' '}
              <Link 
                href="/signup" 
                className="font-medium text-primary-600 hover:text-primary-500 transition-colors"
              >
                create a new account
              </Link>
            </p>
          </div>

          {/* Demo Account Info */}
          <div id="demo-info" className={`p-4 rounded-lg border mb-6 ${
            isDark 
              ? 'bg-blue-900 border-blue-800 text-blue-200' 
              : 'bg-blue-50 border-blue-200 text-blue-700'
          }`}>
            <p className="text-sm font-medium mb-1">Demo Account</p>
            <p className="text-xs">Email: demo@example.com | Password: password</p>
          </div>

          {/* Login Form */}
          <form id="login-form" onSubmit={handleSubmit} className="space-y-6">
            {/* Email Field */}
            <div id="email-field">
              <label 
                htmlFor="email"
                id="email-label" 
                className={`block text-sm font-medium mb-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}
              >
                Email address
              </label>
              <div id="email-input-container" className="relative">
                <Mail className={`absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 ${isDark ? 'text-gray-500' : 'text-gray-400'}`} />
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className={`w-full pl-10 pr-3 py-3 border rounded-lg transition-colors focus:ring-2 focus:ring-primary-500 focus:border-primary-500 ${
                    isDark 
                      ? 'bg-gray-800 border-gray-700 text-white placeholder-gray-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                  }`}
                  placeholder="Enter your email"
                  required
                />
              </div>
            </div>

            {/* Password Field */}
            <div id="password-field">
              <label 
                htmlFor="password"
                id="password-label" 
                className={`block text-sm font-medium mb-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}
              >
                Password
              </label>
              <div id="password-input-container" className="relative">
                <Lock className={`absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 ${isDark ? 'text-gray-500' : 'text-gray-400'}`} />
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className={`w-full pl-10 pr-10 py-3 border rounded-lg transition-colors focus:ring-2 focus:ring-primary-500 focus:border-primary-500 ${
                    isDark 
                      ? 'bg-gray-800 border-gray-700 text-white placeholder-gray-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                  }`}
                  placeholder="Enter your password"
                  required
                />
                <button
                  id="toggle-password"
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className={`absolute right-3 top-1/2 transform -translate-y-1/2 ${isDark ? 'text-gray-500 hover:text-gray-400' : 'text-gray-400 hover:text-gray-600'} transition-colors`}
                >
                  {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>
            </div>

            {/* Options */}
            <div id="login-options" className="flex items-center justify-between">
              <div id="remember-me" className="flex items-center">
                <input
                  id="remember-checkbox"
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)}
                  className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                />
                <label 
                  htmlFor="remember-checkbox" 
                  id="remember-label"
                  className={`ml-2 block text-sm ${isDark ? 'text-gray-300' : 'text-gray-700'}`}
                >
                  Remember me
                </label>
              </div>
              <Link
                href="/forgot-password"
                id="forgot-password-link"
                className="text-sm font-medium text-primary-600 hover:text-primary-500 transition-colors"
              >
                Forgot password?
              </Link>
            </div>

            {/* Submit Button */}
            <button
              id="login-submit"
              type="submit"
              disabled={isLoading}
              className="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-lg text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? (
                <div id="login-loading" className="flex items-center">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Signing in...
                </div>
              ) : (
                <div id="login-ready" className="flex items-center">
                  Sign in
                  <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
                </div>
              )}
            </button>
          </form>

          {/* Divider */}
          <div id="divider" className="mt-6">
            <div className="relative">
              <div className={`absolute inset-0 flex items-center ${isDark ? 'text-gray-600' : 'text-gray-300'}`}>
                <div className="w-full border-t border-current"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span id="divider-text" className={`px-2 ${isDark ? 'bg-gray-900 text-gray-400' : 'bg-gray-50 text-gray-500'}`}>
                  Or continue with
                </span>
              </div>
            </div>
          </div>

          {/* Social Login */}
          <div id="social-login" className="mt-6 grid grid-cols-2 gap-3">
            <button
              id="google-login"
              onClick={() => handleSocialLogin('google')}
              className={`w-full inline-flex justify-center py-3 px-4 border rounded-lg shadow-sm text-sm font-medium transition-colors ${
                isDark
                  ? 'border-gray-700 text-gray-300 bg-gray-800 hover:bg-gray-700'
                  : 'border-gray-300 text-gray-500 bg-white hover:bg-gray-50'
              }`}
            >
              <Google className="h-5 w-5" />
              <span className="ml-2">Google</span>
            </button>
            <button
              id="github-login"
              onClick={() => handleSocialLogin('github')}
              className={`w-full inline-flex justify-center py-3 px-4 border rounded-lg shadow-sm text-sm font-medium transition-colors ${
                isDark
                  ? 'border-gray-700 text-gray-300 bg-gray-800 hover:bg-gray-700'
                  : 'border-gray-300 text-gray-500 bg-white hover:bg-gray-50'
              }`}
            >
              <Github className="h-5 w-5" />
              <span className="ml-2">GitHub</span>
            </button>
          </div>

          {/* Footer */}
          <div id="login-footer" className={`mt-8 text-center text-xs ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
            <p>
              By signing in, you agree to our{' '}
              <Link href="/terms" className="text-primary-600 hover:text-primary-500 transition-colors">
                Terms of Service
              </Link>{' '}
              and{' '}
              <Link href="/privacy" className="text-primary-600 hover:text-primary-500 transition-colors">
                Privacy Policy
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}