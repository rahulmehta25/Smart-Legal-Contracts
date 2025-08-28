/**
 * Waveform Visualizer Component
 * Real-time audio waveform visualization with accessibility support
 * and customizable appearance for voice interface feedback
 */

import React, { useEffect, useRef, useCallback, useState } from 'react';
import { IWaveformVisualizerProps } from '@/types/voice';

export const WaveformVisualizer: React.FC<IWaveformVisualizerProps> = ({
  audioData,
  width,
  height,
  color = '#3b82f6',
  backgroundColor = 'transparent',
  animated = true,
  className = '',
  id = 'waveform-visualizer'
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const [isActive, setIsActive] = useState(false);
  
  // Configuration
  const config = {
    barCount: Math.min(64, Math.floor(width / 4)),
    minBarHeight: 2,
    maxBarHeight: height * 0.8,
    barSpacing: 2,
    smoothing: 0.7,
    sensitivity: 1.5,
    falloffRate: 0.95
  };

  // Previous frame data for smoothing
  const previousData = useRef<Float32Array>(new Float32Array(config.barCount));
  const falloffData = useRef<Float32Array>(new Float32Array(config.barCount));

  /**
   * Process audio data for visualization
   */
  const processAudioData = useCallback((data: Float32Array): Float32Array => {
    if (!data || data.length === 0) {
      return new Float32Array(config.barCount);
    }

    const processedData = new Float32Array(config.barCount);
    const dataSliceSize = Math.floor(data.length / config.barCount);

    for (let i = 0; i < config.barCount; i++) {
      const start = i * dataSliceSize;
      const end = start + dataSliceSize;
      let sum = 0;

      // Calculate average amplitude for this frequency range
      for (let j = start; j < end && j < data.length; j++) {
        sum += Math.abs(data[j]);
      }

      const average = sum / dataSliceSize;
      
      // Apply sensitivity and normalize
      processedData[i] = Math.min(1, average * config.sensitivity);
    }

    return processedData;
  }, [config.barCount, config.sensitivity]);

  /**
   * Apply smoothing to prevent jarring transitions
   */
  const applySmooothing = useCallback((newData: Float32Array): Float32Array => {
    const smoothedData = new Float32Array(config.barCount);

    for (let i = 0; i < config.barCount; i++) {
      // Apply exponential smoothing
      smoothedData[i] = 
        config.smoothing * previousData.current[i] + 
        (1 - config.smoothing) * newData[i];
      
      // Apply falloff for more natural decay
      falloffData.current[i] = Math.max(
        smoothedData[i],
        falloffData.current[i] * config.falloffRate
      );
      
      smoothedData[i] = falloffData.current[i];
    }

    // Update previous data
    previousData.current.set(smoothedData);

    return smoothedData;
  }, [config.smoothing, config.falloffRate, config.barCount]);

  /**
   * Draw waveform on canvas
   */
  const drawWaveform = useCallback((data: Float32Array) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Set up drawing parameters
    const barWidth = (width - (config.barCount - 1) * config.barSpacing) / config.barCount;
    const centerY = height / 2;

    // Set styles
    ctx.fillStyle = color;

    // Check if there's any activity
    const hasActivity = data.some(value => value > 0.01);
    setIsActive(hasActivity);

    if (!hasActivity && !animated) {
      // Draw flat line when no activity
      ctx.fillRect(0, centerY - 1, width, 2);
      return;
    }

    // Draw bars
    for (let i = 0; i < config.barCount; i++) {
      const barHeight = Math.max(
        config.minBarHeight,
        data[i] * config.maxBarHeight
      );

      const x = i * (barWidth + config.barSpacing);
      const y = centerY - barHeight / 2;

      // Add gradient for visual appeal
      if (animated && hasActivity) {
        const gradient = ctx.createLinearGradient(x, y, x, y + barHeight);
        gradient.addColorStop(0, color);
        gradient.addColorStop(1, color + '80'); // Add transparency
        ctx.fillStyle = gradient;
      } else {
        ctx.fillStyle = color;
      }

      // Draw rounded rectangle
      this.drawRoundedRect(ctx, x, y, barWidth, barHeight, 2);
    }
  }, [width, height, color, animated, config]);

  /**
   * Draw rounded rectangle
   */
  const drawRoundedRect = useCallback((
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    width: number,
    height: number,
    radius: number
  ) => {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
    ctx.fill();
  }, []);

  /**
   * Animation loop
   */
  const animate = useCallback(() => {
    if (!audioData || audioData.length === 0) {
      // Draw idle animation when no audio data
      const idleData = new Float32Array(config.barCount);
      for (let i = 0; i < config.barCount; i++) {
        idleData[i] = 0.1 + 0.05 * Math.sin(Date.now() * 0.003 + i * 0.5);
      }
      drawWaveform(idleData);
    } else {
      const processedData = processAudioData(audioData);
      const smoothedData = applySmooothing(processedData);
      drawWaveform(smoothedData);
    }

    if (animated) {
      animationRef.current = requestAnimationFrame(animate);
    }
  }, [audioData, animated, processAudioData, applySmooothing, drawWaveform, config.barCount]);

  /**
   * Start animation
   */
  const startAnimation = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    animate();
  }, [animate]);

  /**
   * Stop animation
   */
  const stopAnimation = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = undefined;
    }
  }, []);

  // Set up canvas and start animation
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Set canvas size
    canvas.width = width;
    canvas.height = height;

    // Set up high DPI support
    const ctx = canvas.getContext('2d');
    if (ctx) {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.scale(dpr, dpr);
    }

    if (animated) {
      startAnimation();
    } else {
      // Single frame render
      const processedData = processAudioData(audioData);
      drawWaveform(processedData);
    }

    return () => {
      stopAnimation();
    };
  }, [width, height, animated, audioData, startAnimation, stopAnimation, processAudioData, drawWaveform]);

  // Handle audio data changes
  useEffect(() => {
    if (!animated) {
      const processedData = processAudioData(audioData);
      const smoothedData = applySmooothing(processedData);
      drawWaveform(smoothedData);
    }
  }, [audioData, animated, processAudioData, applySmooothing, drawWaveform]);

  /**
   * Get accessibility description
   */
  const getAccessibilityDescription = () => {
    if (!audioData || audioData.length === 0) {
      return 'Audio visualizer showing no activity';
    }

    const averageLevel = audioData.reduce((sum, val) => sum + Math.abs(val), 0) / audioData.length;
    
    if (averageLevel > 0.5) {
      return 'Audio visualizer showing high activity';
    } else if (averageLevel > 0.1) {
      return 'Audio visualizer showing moderate activity';
    } else {
      return 'Audio visualizer showing low activity';
    }
  };

  return (
    <div 
      className={`waveform-visualizer ${className}`.trim()}
      style={{
        display: 'inline-block',
        position: 'relative',
        background: backgroundColor,
        borderRadius: '4px',
        overflow: 'hidden'
      }}
    >
      <canvas
        ref={canvasRef}
        id={id}
        width={width}
        height={height}
        style={{
          display: 'block',
          width: `${width}px`,
          height: `${height}px`
        }}
        role="img"
        aria-label={getAccessibilityDescription()}
        aria-live="polite"
      />
      
      {/* Activity indicator for screen readers */}
      <div
        className="sr-only"
        aria-live="polite"
        aria-atomic="true"
      >
        {isActive ? 'Voice activity detected' : 'No voice activity'}
      </div>

      {/* Accessibility description */}
      <div
        id={`${id}-description`}
        className="sr-only"
      >
        Real-time audio waveform visualization showing voice input levels and patterns
      </div>

      <style jsx>{`
        /* High contrast mode support */
        @media (prefers-contrast: high) {
          .waveform-visualizer canvas {
            filter: contrast(150%);
          }
        }

        /* Reduced motion support */
        @media (prefers-reduced-motion: reduce) {
          .waveform-visualizer canvas {
            animation: none !important;
          }
        }

        /* Focus indicator for keyboard navigation */
        .waveform-visualizer:focus-within {
          outline: 2px solid #3b82f6;
          outline-offset: 2px;
        }
      `}</style>
    </div>
  );
};

export default WaveformVisualizer;