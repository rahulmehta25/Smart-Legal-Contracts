import React, { useState } from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';
import { Eye, EyeOff, Shield, Mail, Lock, User, ArrowRight, Github, Google, Check } from 'lucide-react';
import { toast } from 'react-hot-toast';
import { useAuth } from '../src/contexts/AuthContext';
import { useTheme } from '../src/contexts/ThemeContext';

interface Plan {
  id: string;
  name: string;
  price: string;
  description: string;
  features: string[];
  popular?: boolean;
}

const plans: Plan[] = [
  {
    id: 'free',
    name: 'Free',
    price: '$0',
    description: 'Perfect for trying out our service',
    features: [
      '10 documents per month',
      'Basic analysis reports',
      'Email support',
      'Standard processing speed'
    ]
  },
  {
    id: 'pro',
    name: 'Professional',
    price: '$29',
    description: 'Best for small to medium businesses',
    features: [
      '100 documents per month',
      'Advanced analysis & insights',
      'Priority support',
      'Fast processing',
      'Export capabilities',
      'Team collaboration'
    ],
    popular: true
  },
  {
    id: 'enterprise',
    name: 'Enterprise',
    price: '$99',
    description: 'For large organizations',
    features: [
      'Unlimited documents',
      'Custom integrations',
      'Dedicated support',
      'Instant processing',
      'Advanced security',
      'API access',
      'White-label options'
    ]
  }
];

export default function SignupPage() {
  const router = useRouter();
  const { register } = useAuth();
  const { isDark } = useTheme();
  
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedPlan, setSelectedPlan] = useState('pro');
  const [agreedToTerms, setAgreedToTerms] = useState(false);
  const [passwordStrength, setPasswordStrength] = useState(0);

  const calculatePasswordStrength = (password: string): number => {
    let strength = 0;
    if (password.length >= 8) strength++;
    if (/[a-z]/.test(password)) strength++;
    if (/[A-Z]/.test(password)) strength++;
    if (/\d/.test(password)) strength++;
    if (/[^A-Za-z0-9]/.test(password)) strength++;
    return strength;
  };

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    
    if (field === 'password') {
      setPasswordStrength(calculatePasswordStrength(value));
    }
  };

  const validateForm = (): boolean => {
    if (!formData.name.trim()) {
      toast.error('Please enter your name');
      return false;
    }

    if (!formData.email.trim()) {
      toast.error('Please enter your email');
      return false;
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      toast.error('Please enter a valid email address');
      return false;
    }

    if (formData.password.length < 8) {
      toast.error('Password must be at least 8 characters long');
      return false;
    }

    if (formData.password !== formData.confirmPassword) {
      toast.error('Passwords do not match');
      return false;
    }

    if (!agreedToTerms) {
      toast.error('Please agree to the Terms of Service and Privacy Policy');
      return false;
    }

    return true;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) return;

    setIsLoading(true);
    
    try {
      await register(formData.name, formData.email, formData.password);
      toast.success('Account created successfully!');
      router.push('/dashboard');
    } catch (error) {
      toast.error('Failed to create account. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSocialSignup = async (provider: 'google' | 'github') => {
    try {
      toast.loading(`Connecting to ${provider}...`);
      await new Promise(resolve => setTimeout(resolve, 2000));
      toast.success(`Successfully connected to ${provider}!`);
      router.push('/dashboard');
    } catch (error) {
      toast.error(`Failed to connect to ${provider}`);
    }
  };

  const getPasswordStrengthColor = (): string => {
    switch (passwordStrength) {
      case 0:
      case 1:
        return 'bg-red-500';
      case 2:
        return 'bg-orange-500';
      case 3:
        return 'bg-yellow-500';
      case 4:
        return 'bg-blue-500';
      case 5:
        return 'bg-green-500';
      default:
        return 'bg-gray-300';
    }
  };

  const getPasswordStrengthText = (): string => {
    switch (passwordStrength) {
      case 0:
      case 1:
        return 'Very Weak';
      case 2:
        return 'Weak';
      case 3:
        return 'Fair';
      case 4:
        return 'Good';
      case 5:
        return 'Strong';
      default:
        return '';
    }
  };

  return (
    <div id="signup-container" className={`min-h-screen ${isDark ? 'bg-gray-900' : 'bg-gray-50'}`}>
      {/* Header */}
      <div id="signup-header" className={`${isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-b`}>
        <div id="header-content" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div id="header-inner" className="flex items-center justify-between h-16">
            <Link href="/" id="header-logo" className="flex items-center space-x-3">
              <div id="logo-icon" className="p-2 bg-primary-100 rounded-lg">
                <Shield className="h-6 w-6 text-primary-600" />
              </div>
              <span id="brand-name" className={`text-xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                ArbiScan Pro
              </span>
            </Link>
            
            <div id="header-actions" className="flex items-center space-x-4">
              <span id="existing-account-text" className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                Already have an account?
              </span>
              <Link
                href="/login"
                id="login-link"
                className="text-sm font-medium text-primary-600 hover:text-primary-500 transition-colors"
              >
                Sign in
              </Link>
            </div>
          </div>
        </div>
      </div>

      <div id="signup-content" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div id="signup-layout" className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          {/* Left Side - Form */}
          <div id="form-section">
            <div id="form-header" className="mb-8">
              <h1 id="form-title" className={`text-3xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Create your account
              </h1>
              <p id="form-description" className={`mt-2 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                Get started with ArbiScan Pro today and revolutionize your legal document analysis.
              </p>
            </div>

            {/* Signup Form */}
            <form id="signup-form" onSubmit={handleSubmit} className="space-y-6">
              {/* Name Field */}
              <div id="name-field">
                <label 
                  htmlFor="name"
                  id="name-label" 
                  className={`block text-sm font-medium mb-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}
                >
                  Full Name
                </label>
                <div id="name-input-container" className="relative">
                  <User className={`absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 ${isDark ? 'text-gray-500' : 'text-gray-400'}`} />
                  <input
                    id="name"
                    type="text"
                    value={formData.name}
                    onChange={(e) => handleInputChange('name', e.target.value)}
                    className={`w-full pl-10 pr-3 py-3 border rounded-lg transition-colors focus:ring-2 focus:ring-primary-500 focus:border-primary-500 ${
                      isDark 
                        ? 'bg-gray-800 border-gray-700 text-white placeholder-gray-400' 
                        : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                    }`}
                    placeholder="Enter your full name"
                    required
                  />
                </div>
              </div>

              {/* Email Field */}
              <div id="email-field">
                <label 
                  htmlFor="email"
                  id="email-label" 
                  className={`block text-sm font-medium mb-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}
                >
                  Email Address
                </label>
                <div id="email-input-container" className="relative">
                  <Mail className={`absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 ${isDark ? 'text-gray-500' : 'text-gray-400'}`} />
                  <input
                    id="email"
                    type="email"
                    value={formData.email}
                    onChange={(e) => handleInputChange('email', e.target.value)}
                    className={`w-full pl-10 pr-3 py-3 border rounded-lg transition-colors focus:ring-2 focus:ring-primary-500 focus:border-primary-500 ${
                      isDark 
                        ? 'bg-gray-800 border-gray-700 text-white placeholder-gray-400' 
                        : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                    }`}
                    placeholder="Enter your email address"
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
                    value={formData.password}
                    onChange={(e) => handleInputChange('password', e.target.value)}
                    className={`w-full pl-10 pr-10 py-3 border rounded-lg transition-colors focus:ring-2 focus:ring-primary-500 focus:border-primary-500 ${
                      isDark 
                        ? 'bg-gray-800 border-gray-700 text-white placeholder-gray-400' 
                        : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                    }`}
                    placeholder="Create a strong password"
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
                
                {/* Password Strength Indicator */}
                {formData.password && (
                  <div id="password-strength" className="mt-2">
                    <div className="flex items-center space-x-2">
                      <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div 
                          className={`h-full transition-all duration-300 ${getPasswordStrengthColor()}`}
                          style={{ width: `${(passwordStrength / 5) * 100}%` }}
                        ></div>
                      </div>
                      <span className={`text-xs font-medium ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                        {getPasswordStrengthText()}
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Confirm Password Field */}
              <div id="confirm-password-field">
                <label 
                  htmlFor="confirmPassword"
                  id="confirm-password-label" 
                  className={`block text-sm font-medium mb-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}
                >
                  Confirm Password
                </label>
                <div id="confirm-password-input-container" className="relative">
                  <Lock className={`absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 ${isDark ? 'text-gray-500' : 'text-gray-400'}`} />
                  <input
                    id="confirmPassword"
                    type={showConfirmPassword ? 'text' : 'password'}
                    value={formData.confirmPassword}
                    onChange={(e) => handleInputChange('confirmPassword', e.target.value)}
                    className={`w-full pl-10 pr-10 py-3 border rounded-lg transition-colors focus:ring-2 focus:ring-primary-500 focus:border-primary-500 ${
                      isDark 
                        ? 'bg-gray-800 border-gray-700 text-white placeholder-gray-400' 
                        : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                    }`}
                    placeholder="Confirm your password"
                    required
                  />
                  <button
                    id="toggle-confirm-password"
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className={`absolute right-3 top-1/2 transform -translate-y-1/2 ${isDark ? 'text-gray-500 hover:text-gray-400' : 'text-gray-400 hover:text-gray-600'} transition-colors`}
                  >
                    {showConfirmPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                  </button>
                </div>
              </div>

              {/* Terms Agreement */}
              <div id="terms-agreement" className="flex items-start space-x-3">
                <input
                  id="terms-checkbox"
                  type="checkbox"
                  checked={agreedToTerms}
                  onChange={(e) => setAgreedToTerms(e.target.checked)}
                  className="mt-1 h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                />
                <label 
                  htmlFor="terms-checkbox" 
                  id="terms-label"
                  className={`text-sm ${isDark ? 'text-gray-300' : 'text-gray-700'}`}
                >
                  I agree to the{' '}
                  <Link href="/terms" className="text-primary-600 hover:text-primary-500 transition-colors">
                    Terms of Service
                  </Link>{' '}
                  and{' '}
                  <Link href="/privacy" className="text-primary-600 hover:text-primary-500 transition-colors">
                    Privacy Policy
                  </Link>
                </label>
              </div>

              {/* Submit Button */}
              <button
                id="signup-submit"
                type="submit"
                disabled={isLoading}
                className="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-lg text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isLoading ? (
                  <div id="signup-loading" className="flex items-center">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Creating account...
                  </div>
                ) : (
                  <div id="signup-ready" className="flex items-center">
                    Create Account
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
                    Or sign up with
                  </span>
                </div>
              </div>
            </div>

            {/* Social Signup */}
            <div id="social-signup" className="mt-6 grid grid-cols-2 gap-3">
              <button
                id="google-signup"
                onClick={() => handleSocialSignup('google')}
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
                id="github-signup"
                onClick={() => handleSocialSignup('github')}
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
          </div>

          {/* Right Side - Pricing Plans */}
          <div id="pricing-section">
            <div id="pricing-header" className="mb-8">
              <h2 id="pricing-title" className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Choose Your Plan
              </h2>
              <p id="pricing-description" className={`mt-2 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                Start with any plan and upgrade anytime as your needs grow.
              </p>
            </div>

            <div id="pricing-plans" className="space-y-4">
              {plans.map((plan) => (
                <div
                  key={plan.id}
                  id={`plan-${plan.id}`}
                  className={`relative p-6 border rounded-lg cursor-pointer transition-all ${
                    selectedPlan === plan.id
                      ? 'border-primary-500 bg-primary-50 ring-2 ring-primary-500'
                      : isDark
                      ? 'border-gray-700 bg-gray-800 hover:border-gray-600'
                      : 'border-gray-300 bg-white hover:border-gray-400'
                  }`}
                  onClick={() => setSelectedPlan(plan.id)}
                >
                  {plan.popular && (
                    <div id={`plan-${plan.id}-popular`} className="absolute -top-3 left-4">
                      <span className="bg-primary-600 text-white px-3 py-1 text-xs font-medium rounded-full">
                        Most Popular
                      </span>
                    </div>
                  )}
                  
                  <div id={`plan-${plan.id}-header`} className="flex items-center justify-between mb-4">
                    <div>
                      <h3 id={`plan-${plan.id}-name`} className={`text-lg font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                        {plan.name}
                      </h3>
                      <p id={`plan-${plan.id}-price`} className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                        {plan.price}
                        {plan.id !== 'free' && <span className={`text-sm font-normal ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>/month</span>}
                      </p>
                    </div>
                    <div id={`plan-${plan.id}-radio`} className={`w-6 h-6 rounded-full border-2 flex items-center justify-center ${
                      selectedPlan === plan.id
                        ? 'border-primary-500 bg-primary-500'
                        : isDark
                        ? 'border-gray-600'
                        : 'border-gray-300'
                    }`}>
                      {selectedPlan === plan.id && (
                        <div className="w-2 h-2 bg-white rounded-full"></div>
                      )}
                    </div>
                  </div>
                  
                  <p id={`plan-${plan.id}-description`} className={`text-sm mb-4 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                    {plan.description}
                  </p>
                  
                  <ul id={`plan-${plan.id}-features`} className="space-y-2">
                    {plan.features.map((feature, index) => (
                      <li key={index} id={`plan-${plan.id}-feature-${index}`} className="flex items-center">
                        <Check className="h-4 w-4 text-green-500 mr-2 flex-shrink-0" />
                        <span className={`text-sm ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
                          {feature}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>

            <div id="pricing-note" className={`mt-6 p-4 rounded-lg ${isDark ? 'bg-gray-800 border-gray-700' : 'bg-blue-50 border-blue-200'} border`}>
              <p className={`text-sm ${isDark ? 'text-gray-300' : 'text-blue-700'}`}>
                <strong>30-day free trial</strong> on all paid plans. No credit card required to start.
                Cancel anytime.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}