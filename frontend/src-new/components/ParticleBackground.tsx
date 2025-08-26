import { useEffect, useRef } from 'react';
import { Scale, FileText, Gavel, BookOpen } from 'lucide-react';

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  opacity: number;
  icon: number;
  rotation: number;
  rotationSpeed: number;
}

export const ParticleBackground = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const animationRef = useRef<number>();

  const icons = [Scale, FileText, Gavel, BookOpen];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    const createParticle = (): Particle => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      size: Math.random() * 20 + 10,
      opacity: Math.random() * 0.3 + 0.1,
      icon: Math.floor(Math.random() * icons.length),
      rotation: Math.random() * 360,
      rotationSpeed: (Math.random() - 0.5) * 2,
    });

    const initParticles = () => {
      particlesRef.current = Array.from({ length: 15 }, createParticle);
    };

    const drawParticle = (particle: Particle) => {
      ctx.save();
      ctx.translate(particle.x, particle.y);
      ctx.rotate((particle.rotation * Math.PI) / 180);
      ctx.globalAlpha = particle.opacity;
      
      // Draw legal icon as simple shapes
      ctx.strokeStyle = '#ffd700';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      switch (particle.icon) {
        case 0: // Scale
          ctx.arc(0, 0, particle.size / 2, 0, Math.PI * 2);
          ctx.moveTo(-particle.size / 2, 0);
          ctx.lineTo(particle.size / 2, 0);
          break;
        case 1: // Document
          ctx.rect(-particle.size / 3, -particle.size / 2, particle.size * 2/3, particle.size);
          break;
        case 2: // Gavel
          ctx.rect(-particle.size / 2, -particle.size / 4, particle.size, particle.size / 2);
          break;
        case 3: // Book
          ctx.rect(-particle.size / 2, -particle.size / 3, particle.size, particle.size * 2/3);
          break;
      }
      
      ctx.stroke();
      ctx.restore();
    };

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particlesRef.current.forEach((particle) => {
        particle.x += particle.vx;
        particle.y += particle.vy;
        particle.rotation += particle.rotationSpeed;

        // Wrap around edges
        if (particle.x < -50) particle.x = canvas.width + 50;
        if (particle.x > canvas.width + 50) particle.x = -50;
        if (particle.y < -50) particle.y = canvas.height + 50;
        if (particle.y > canvas.height + 50) particle.y = -50;

        drawParticle(particle);
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    resizeCanvas();
    initParticles();
    animate();

    window.addEventListener('resize', resizeCanvas);

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none opacity-30"
      style={{ zIndex: -1 }}
    />
  );
};