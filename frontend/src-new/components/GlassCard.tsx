import { ReactNode, useState } from 'react';
import { cn } from '@/lib/utils';

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  hover3d?: boolean;
  magnetic?: boolean;
}

export const GlassCard = ({ 
  children, 
  className = "",
  hover3d = false,
  magnetic = false
}: GlassCardProps) => {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isHovered, setIsHovered] = useState(false);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!hover3d) return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    
    const rotateX = (y - centerY) / 10;
    const rotateY = (centerX - x) / 10;
    
    setMousePosition({ x: rotateY, y: rotateX });
  };

  const handleMouseLeave = () => {
    setMousePosition({ x: 0, y: 0 });
    setIsHovered(false);
  };

  const handleMouseEnter = () => {
    setIsHovered(true);
  };

  return (
    <div
      className={cn(
        "glass-card-hover rounded-xl p-6 transition-all duration-500",
        hover3d && "card-3d",
        magnetic && "btn-magnetic",
        className
      )}
      style={hover3d ? {
        transform: `perspective(1000px) rotateX(${mousePosition.y}deg) rotateY(${mousePosition.x}deg) scale(${isHovered ? 1.05 : 1})`,
      } : undefined}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onMouseEnter={handleMouseEnter}
    >
      {children}
    </div>
  );
};