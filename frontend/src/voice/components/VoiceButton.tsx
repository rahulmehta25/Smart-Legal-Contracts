/**
 * Voice Button Component
 * Interactive voice activation button with visual feedback, accessibility support,
 * and real-time state indication
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { IVoiceButtonProps } from '@/types/voice';

export const VoiceButton: React.FC<IVoiceButtonProps> = ({
  isListening,
  isProcessing,
  disabled = false,
  size = 'medium',
  variant = 'primary',
  onStartListening,
  onStopListening,
  className = '',
  id = 'voice-button'
}) => {
  const [ripple, setRipple] = useState<{ x: number; y: number; size: number } | null>(null);
  const [pulseAnimation, setPulseAnimation] = useState(false);
  const buttonRef = useRef<HTMLButtonElement>(null);

  // Size configurations
  const sizeConfig = {
    small: {
      width: '40px',
      height: '40px',
      iconSize: '18px',
      fontSize: '14px'
    },
    medium: {
      width: '56px',
      height: '56px',
      iconSize: '24px',
      fontSize: '16px'
    },
    large: {
      width: '72px',
      height: '72px',
      iconSize: '32px',
      fontSize: '18px'
    }
  };

  const currentSize = sizeConfig[size];

  // Color configurations
  const variantConfig = {
    primary: {
      background: isListening ? '#ef4444' : '#3b82f6',
      backgroundHover: isListening ? '#dc2626' : '#2563eb',
      color: '#ffffff',
      border: 'none',
      shadow: '0 4px 14px rgba(59, 130, 246, 0.25)'
    },
    secondary: {
      background: isListening ? '#ef4444' : '#6b7280',
      backgroundHover: isListening ? '#dc2626' : '#4b5563',
      color: '#ffffff',
      border: 'none',
      shadow: '0 4px 14px rgba(107, 114, 128, 0.25)'
    },
    danger: {
      background: '#ef4444',
      backgroundHover: '#dc2626',
      color: '#ffffff',
      border: 'none',
      shadow: '0 4px 14px rgba(239, 68, 68, 0.25)'
    }
  };

  const currentVariant = variantConfig[variant];

  // Handle click with ripple effect
  const handleClick = useCallback((event: React.MouseEvent<HTMLButtonElement>) => {
    if (disabled || isProcessing) return;

    const button = event.currentTarget;
    const rect = button.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = event.clientX - rect.left - size / 2;
    const y = event.clientY - rect.top - size / 2;

    setRipple({ x, y, size });

    // Clear ripple after animation
    setTimeout(() => setRipple(null), 600);

    // Toggle listening state
    if (isListening) {
      onStopListening();
    } else {
      onStartListening();
    }
  }, [disabled, isProcessing, isListening, onStartListening, onStopListening]);

  // Handle keyboard interaction
  const handleKeyDown = useCallback((event: React.KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === ' ' || event.key === 'Enter') {
      event.preventDefault();
      if (!disabled && !isProcessing) {
        if (isListening) {
          onStopListening();
        } else {
          onStartListening();
        }
      }
    }
  }, [disabled, isProcessing, isListening, onStartListening, onStopListening]);

  // Pulse animation for listening state
  useEffect(() => {
    if (isListening) {
      setPulseAnimation(true);
      const interval = setInterval(() => {
        setPulseAnimation(prev => !prev);
      }, 1000);
      return () => clearInterval(interval);
    } else {
      setPulseAnimation(false);
    }
  }, [isListening]);

  // Get button text and ARIA labels
  const getButtonText = () => {
    if (isProcessing) return 'Processing...';
    if (isListening) return 'Stop Listening';
    return 'Start Voice Input';
  };

  const getAriaLabel = () => {
    if (disabled) return 'Voice input disabled';
    if (isProcessing) return 'Processing voice input';
    if (isListening) return 'Stop voice input (currently listening)';
    return 'Start voice input';
  };

  // Generate microphone icon SVG
  const MicrophoneIcon = () => (
    <svg
      width={currentSize.iconSize}
      height={currentSize.iconSize}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      style={{ transition: 'all 0.2s ease' }}
    >
      {isListening ? (
        // Stop icon when listening
        <rect
          x="6"
          y="6"
          width="12"
          height="12"
          rx="2"
          fill="currentColor"
        />
      ) : (
        // Microphone icon when not listening
        <>
          <path
            d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4z"
            fill="currentColor"
          />
          <path
            d="M19 11v1a7 7 0 0 1-14 0v-1M12 19v4M8 23h8"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            fill="none"
          />
        </>
      )}
    </svg>
  );

  // Generate processing/loading icon
  const LoadingIcon = () => (
    <svg
      width={currentSize.iconSize}
      height={currentSize.iconSize}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      style={{
        animation: 'spin 1s linear infinite',
        transformOrigin: 'center'
      }}
    >
      <circle
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
        strokeDasharray="31.416"
        strokeDashoffset="15.708"
        fill="none"
        strokeLinecap="round"
      />
    </svg>
  );

  return (
    <div className="voice-button-container" style={{ display: 'inline-block', position: 'relative' }}>
      <button
        ref={buttonRef}
        id={id}
        className={`voice-button ${className}`.trim()}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        aria-label={getAriaLabel()}
        aria-pressed={isListening}
        aria-live="polite"
        style={{
          position: 'relative',
          width: currentSize.width,
          height: currentSize.height,
          borderRadius: '50%',
          border: currentVariant.border,
          background: currentVariant.background,
          color: currentVariant.color,
          cursor: disabled ? 'not-allowed' : 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: currentSize.fontSize,
          fontWeight: '600',
          boxShadow: disabled ? 'none' : currentVariant.shadow,
          opacity: disabled ? 0.6 : 1,
          overflow: 'hidden',
          transition: 'all 0.2s ease, transform 0.1s ease',
          transform: isListening && pulseAnimation ? 'scale(1.1)' : 'scale(1)',
          ...(isListening && {
            boxShadow: `0 0 20px rgba(239, 68, 68, 0.5), ${currentVariant.shadow}`,
            animation: 'pulse 1s infinite'
          })
        }}
        onMouseEnter={(e) => {
          if (!disabled) {
            e.currentTarget.style.background = currentVariant.backgroundHover;
            e.currentTarget.style.transform = 'scale(1.05)';
          }
        }}
        onMouseLeave={(e) => {
          if (!disabled) {
            e.currentTarget.style.background = currentVariant.background;
            e.currentTarget.style.transform = isListening && pulseAnimation ? 'scale(1.1)' : 'scale(1)';
          }
        }}
        onMouseDown={(e) => {
          if (!disabled) {
            e.currentTarget.style.transform = 'scale(0.95)';
          }
        }}
        onMouseUp={(e) => {
          if (!disabled) {
            e.currentTarget.style.transform = isListening && pulseAnimation ? 'scale(1.1)' : 'scale(1.05)';
          }
        }}
      >
        {/* Icon */}
        <div style={{ zIndex: 2, position: 'relative' }}>
          {isProcessing ? <LoadingIcon /> : <MicrophoneIcon />}
        </div>

        {/* Ripple effect */}
        {ripple && (
          <div
            style={{
              position: 'absolute',
              left: ripple.x,
              top: ripple.y,
              width: ripple.size,
              height: ripple.size,
              borderRadius: '50%',
              background: 'rgba(255, 255, 255, 0.3)',
              animation: 'ripple 0.6s linear',
              pointerEvents: 'none',
              zIndex: 1
            }}
          />
        )}

        {/* Listening indicator rings */}
        {isListening && (
          <>
            <div
              style={{
                position: 'absolute',
                top: '-8px',
                left: '-8px',
                right: '-8px',
                bottom: '-8px',
                border: '2px solid rgba(239, 68, 68, 0.3)',
                borderRadius: '50%',
                animation: 'pulse-ring 1s infinite',
                pointerEvents: 'none'
              }}
            />
            <div
              style={{
                position: 'absolute',
                top: '-16px',
                left: '-16px',
                right: '-16px',
                bottom: '-16px',
                border: '2px solid rgba(239, 68, 68, 0.2)',
                borderRadius: '50%',
                animation: 'pulse-ring 1s infinite 0.5s',
                pointerEvents: 'none'
              }}
            />
          </>
        )}
      </button>

      {/* Status text */}
      <div
        style={{
          position: 'absolute',
          top: '100%',
          left: '50%',
          transform: 'translateX(-50%)',
          marginTop: '8px',
          fontSize: '12px',
          fontWeight: '500',
          color: isListening ? '#ef4444' : '#6b7280',
          whiteSpace: 'nowrap',
          opacity: isListening || isProcessing ? 1 : 0,
          transition: 'opacity 0.2s ease'
        }}
        aria-live="polite"
      >
        {getButtonText()}
      </div>

      {/* CSS Animations */}
      <style jsx>{`
        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.8;
          }
        }

        @keyframes pulse-ring {
          0% {
            transform: scale(1);
            opacity: 1;
          }
          100% {
            transform: scale(1.2);
            opacity: 0;
          }
        }

        @keyframes ripple {
          0% {
            transform: scale(0);
            opacity: 0.7;
          }
          100% {
            transform: scale(1);
            opacity: 0;
          }
        }

        @keyframes spin {
          0% {
            transform: rotate(0deg);
          }
          100% {
            transform: rotate(360deg);
          }
        }

        .voice-button:focus {
          outline: 3px solid rgba(59, 130, 246, 0.5);
          outline-offset: 2px;
        }

        .voice-button:active {
          transform: scale(0.95);
        }

        /* High contrast mode support */
        @media (prefers-contrast: high) {
          .voice-button {
            border: 2px solid currentColor !important;
          }
        }

        /* Reduced motion support */
        @media (prefers-reduced-motion: reduce) {
          .voice-button,
          .voice-button *,
          .voice-button::before,
          .voice-button::after {
            animation: none !important;
            transition: none !important;
          }
        }
      `}</style>
    </div>
  );
};

export default VoiceButton;