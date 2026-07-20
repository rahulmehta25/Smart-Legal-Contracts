import React, { useState } from 'react';
import Link from 'next/link';
import { Shield, Mail, ArrowLeft, CheckCircle, AlertCircle } from 'lucide-react';
import { toast } from 'react-hot-toast';
import { useAuth } from '../src/contexts/AuthContext';
import { useTheme } from '../src/contexts/ThemeContext';

export default function ForgotPasswordPage() {
  const { resetPassword } = useAuth();
  const { isDark } = useTheme();
  
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [emailSent, setEmailSent] = useState(false);

  const validateEmail = (email: string): boolean => {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!email) {
      toast.error('Please enter your email address');
      return;
    }

    if (!validateEmail(email)) {
      toast.error('Please enter a valid email address');
      return;
    }

    setIsLoading(true);
    
    try {
      await resetPassword(email);
      setEmailSent(true);
      toast.success('Password reset instructions sent to your email');
    } catch (error) {
      toast.error('Failed to send reset email. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleResendEmail = async () => {
    if (!email) return;
    
    setIsLoading(true);
    try {
      await resetPassword(email);
      toast.success('Reset email sent again');
    } catch (error) {
      toast.error('Failed to resend email');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div id="forgot-password-container" className={`min-h-screen flex ${isDark ? 'bg-gray-900' : 'bg-gray-50'}`}>
      {/* Left Side - Branding (hidden on small screens) */}
      <div id="forgot-password-branding" className={`hidden lg:flex lg:w-1/2 ${isDark ? 'bg-gray-800' : 'bg-primary-600'} flex-col justify-center px-12`}>
        <div id="branding-content" className="max-w-md">
          <div id="logo-section" className="flex items-center space-x-3 mb-8">
            <div id="logo-icon" className="p-3 bg-white bg-opacity-20 rounded-lg">
              <Shield className="h-8 w-8 text-white" />
            </div>
            <h1 id="brand-name" className="text-3xl font-bold text-white">
              ArbiScan Pro
            </h1>
          </div>
          
          <h2 id="branding-title" className="text-4xl font-bold text-white mb-6">
            Secure Account Recovery
          </h2>
          
          <p id="branding-description" className="text-xl text-primary-100 mb-8 leading-relaxed">
            We'll help you regain access to your account quickly and securely. Your data and privacy are our top priority.
          </p>

          {/* Security Features */}
          <div id="security-features" className="space-y-4">
            <div id="security-feature-1" className="flex items-center space-x-3">
              <div className="w-2 h-2 bg-white rounded-full"></div>
              <span className="text-primary-100">256-bit SSL Encryption</span>
            </div>
            <div id="security-feature-2" className="flex items-center space-x-3">
              <div className="w-2 h-2 bg-white rounded-full"></div>
              <span className="text-primary-100">Secure Password Reset Process</span>
            </div>
            <div id="security-feature-3" className="flex items-center space-x-3">
              <div className="w-2 h-2 bg-white rounded-full"></div>
              <span className="text-primary-100">24/7 Security Monitoring</span>
            </div>
            <div id="security-feature-4" className="flex items-center space-x-3">
              <div className="w-2 h-2 bg-white rounded-full"></div>
              <span className="text-primary-100">GDPR Compliant</span>
            </div>
          </div>
        </div>
      </div>

      {/* Right Side - Reset Form */}
      <div id="reset-form-section" className="flex-1 flex flex-col justify-center px-6 sm:px-12 lg:px-16">
        <div id="form-container" className="w-full max-w-md mx-auto">
          {/* Back Link */}
          <div id="back-link-container" className="mb-8">
            <Link 
              href="/login" 
              id="back-link"
              className={`flex items-center space-x-2 text-sm font-medium transition-colors ${
                isDark ? 'text-gray-400 hover:text-gray-300' : 'text-gray-600 hover:text-gray-500'
              }`}
            >
              <ArrowLeft className="h-4 w-4" />
              <span>Back to sign in</span>
            </Link>
          </div>

          {/* Mobile Logo */}
          <div id="mobile-logo" className="lg:hidden text-center mb-8">
            <div id="mobile-logo-icon" className="inline-flex items-center justify-center w-16 h-16 bg-primary-600 rounded-xl mb-4">
              <Shield className="h-8 w-8 text-white" />
            </div>
            <h1 id="mobile-brand-name" className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
              ArbiScan Pro
            </h1>
          </div>

          {!emailSent ? (
            <>
              {/* Form Header */}
              <div id="form-header" className="text-center mb-8">
                <h2 id="form-title" className={`text-3xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                  Forgot your password?
                </h2>
                <p id="form-description" className={`mt-2 text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                  No worries, we'll send you reset instructions via email.
                </p>
              </div>

              {/* Reset Form */}
              <form id="reset-form" onSubmit={handleSubmit} className="space-y-6">
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
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
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

                {/* Submit Button */}
                <button
                  id="reset-submit"
                  type="submit"
                  disabled={isLoading}
                  className="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-lg text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {isLoading ? (
                    <div id="reset-loading" className="flex items-center">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Sending reset email...
                    </div>
                  ) : (
                    <div id="reset-ready" className="flex items-center">
                      <Mail className="mr-2 h-4 w-4" />
                      Send Reset Instructions
                    </div>
                  )}
                </button>
              </form>
            </>
          ) : (
            /* Email Sent Success State */
            <div id="email-sent-container" className="text-center">
              {/* Success Icon */}
              <div id="success-icon-container" className="mb-6">
                <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 rounded-full">
                  <CheckCircle className="h-8 w-8 text-green-600" />
                </div>
              </div>

              {/* Success Message */}
              <div id="success-message" className="mb-8">
                <h2 id="success-title" className={`text-3xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                  Check your email
                </h2>
                <p id="success-description" className={`text-lg mb-2 ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
                  We've sent password reset instructions to:
                </p>
                <p id="email-address" className="text-lg font-semibold text-primary-600 mb-4">
                  {email}
                </p>
                <p id="success-note" className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>
                  If you don't see the email, check your spam folder or try again.
                </p>
              </div>

              {/* Action Buttons */}
              <div id="success-actions" className="space-y-3">
                <button
                  id="resend-button"
                  onClick={handleResendEmail}
                  disabled={isLoading}
                  className={`w-full py-3 px-4 border border-gray-300 rounded-lg text-sm font-medium transition-colors ${
                    isDark
                      ? 'border-gray-600 text-gray-300 bg-gray-800 hover:bg-gray-700 disabled:opacity-50'
                      : 'border-gray-300 text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50'
                  }`}
                >
                  {isLoading ? (
                    <div className="flex items-center justify-center">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mr-2"></div>
                      Resending...
                    </div>
                  ) : (
                    'Resend Email'
                  )}
                </button>
                
                <Link
                  href="/login"
                  id="back-to-login"
                  className="block w-full py-3 px-4 bg-primary-600 text-white text-center rounded-lg hover:bg-primary-700 transition-colors text-sm font-medium"
                >
                  Back to Sign In
                </Link>
              </div>

              {/* Help Section */}
              <div id="help-section" className={`mt-8 p-4 rounded-lg border ${isDark ? 'bg-gray-800 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
                <div className="flex items-start space-x-3">
                  <AlertCircle className={`h-5 w-5 mt-0.5 ${isDark ? 'text-gray-400' : 'text-gray-500'}`} />
                  <div>
                    <h4 id="help-title" className={`text-sm font-medium mb-1 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                      Still need help?
                    </h4>
                    <p id="help-description" className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                      If you're still having trouble accessing your account, please{' '}
                      <Link href="/support" className="text-primary-600 hover:text-primary-500 transition-colors">
                        contact our support team
                      </Link>{' '}
                      for personalized assistance.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Security Notice */}
          <div id="security-notice" className={`mt-8 p-4 rounded-lg border ${isDark ? 'bg-blue-900 border-blue-800' : 'bg-blue-50 border-blue-200'}`}>
            <div className="flex items-start space-x-3">
              <Shield className={`h-5 w-5 mt-0.5 ${isDark ? 'text-blue-400' : 'text-blue-600'}`} />
              <div>
                <h4 id="security-notice-title" className={`text-sm font-medium mb-1 ${isDark ? 'text-blue-200' : 'text-blue-800'}`}>
                  Security Notice
                </h4>
                <p id="security-notice-description" className={`text-sm ${isDark ? 'text-blue-300' : 'text-blue-700'}`}>
                  Password reset links expire after 24 hours for your security. 
                  If someone else requested this reset, you can safely ignore this email.
                </p>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div id="forgot-password-footer" className={`mt-8 text-center text-xs ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
            <p>
              Need immediate help?{' '}
              <Link href="/support" className="text-primary-600 hover:text-primary-500 transition-colors">
                Contact Support
              </Link>{' '}
              or call +1 (555) 123-4567
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}