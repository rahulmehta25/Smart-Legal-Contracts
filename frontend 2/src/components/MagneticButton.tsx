import { ReactNode, useState } from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface MagneticButtonProps {
  children: ReactNode;
  variant?: 'default' | 'hero' | 'legal' | 'neural';
  size?: 'default' | 'sm' | 'lg' | 'icon';
  className?: string;
  onClick?: () => void;
}

export const MagneticButton = ({ 
  children, 
  variant = 'default',
  size = 'default',
  className = "",
  onClick
}: MagneticButtonProps) => {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isHovered, setIsHovered] = useState(false);

  const handleMouseMove = (e: React.MouseEvent<HTMLButtonElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;
    
    setMousePosition({ x: x / 4, y: y / 4 });
  };

  const handleMouseLeave = () => {
    setMousePosition({ x: 0, y: 0 });
    setIsHovered(false);
  };

  const handleMouseEnter = () => {
    setIsHovered(true);
  };

  const variants = {
    hero: "bg-gradient-gold hover:shadow-glow-primary border-primary/20 text-background font-semibold",
    legal: "bg-gradient-legal hover:shadow-glow-accent border-accent/20 text-foreground font-semibold", 
    neural: "bg-gradient-neural hover:shadow-neural border-neural-purple/20 text-foreground font-semibold",
    default: "btn-magnetic"
  };

  return (
    <Button
      variant={variant === 'default' ? 'default' : 'outline'}
      size={size}
      className={cn(
        "btn-magnetic relative overflow-hidden transition-all duration-300 border",
        variants[variant],
        className
      )}
      style={{
        transform: `translate(${mousePosition.x}px, ${mousePosition.y}px) scale(${isHovered ? 1.05 : 1})`,
      }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onMouseEnter={handleMouseEnter}
      onClick={onClick}
    >
      <span className="relative z-10 font-semibold">
        {children}
      </span>
    </Button>
  );
};