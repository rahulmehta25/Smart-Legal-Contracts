import { useEffect, useRef } from 'react';

interface ProcessingNode {
  x: number;
  y: number;
  radius: number;
  active: boolean;
  layer: number;
  connections: number[];
}

export const NeuralNetwork = ({ className = "" }: { className?: string }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<ProcessingNode[]>([]);
  const animationRef = useRef<number>();
  const activationRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resizeCanvas = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };

    const createNodes = () => {
      const nodes: ProcessingNode[] = [];
      const layers = [4, 6, 6, 4]; // Input, Hidden1, Hidden2, Output
      const layerSpacing = canvas.width / (layers.length + 1);
      
      layers.forEach((nodeCount, layerIndex) => {
        const nodeSpacing = canvas.height / (nodeCount + 1);
        
        for (let i = 0; i < nodeCount; i++) {
          const nodeIndex = nodes.length;
          const node: ProcessingNode = {
            x: layerSpacing * (layerIndex + 1),
            y: nodeSpacing * (i + 1),
            radius: layerIndex === 0 || layerIndex === layers.length - 1 ? 8 : 6,
            active: false,
            layer: layerIndex,
            connections: []
          };
          
          // Connect to next layer
          if (layerIndex < layers.length - 1) {
            const nextLayerStart = nodes.length + nodeCount - i;
            const nextLayerSize = layers[layerIndex + 1];
            for (let j = 0; j < nextLayerSize; j++) {
              node.connections.push(nextLayerStart + j);
            }
          }
          
          nodes.push(node);
        }
      });
      
      nodesRef.current = nodes;
    };

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      const nodes = nodesRef.current;
      
      // Simulate processing wave
      activationRef.current += 0.02;
      const wave = Math.sin(activationRef.current);
      
      // Update node activation
      nodes.forEach((node, index) => {
        const layerActivation = (wave + node.layer * 0.5) % (Math.PI * 2);
        node.active = Math.sin(layerActivation) > 0.3;
      });
      
      // Draw connections with smooth gradients
      nodes.forEach((node) => {
        node.connections.forEach(connectionIndex => {
          if (connectionIndex < nodes.length) {
            const targetNode = nodes[connectionIndex];
            
            ctx.beginPath();
            ctx.moveTo(node.x, node.y);
            ctx.lineTo(targetNode.x, targetNode.y);
            
            const intensity = (node.active && targetNode.active) ? 0.8 : 0.2;
            ctx.strokeStyle = `rgba(59, 130, 246, ${intensity})`;
            ctx.lineWidth = intensity > 0.5 ? 2 : 1;
            ctx.stroke();
          }
        });
      });
      
      // Draw nodes with better styling
      nodes.forEach(node => {
        // Outer glow for active nodes
        if (node.active) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.radius + 4, 0, Math.PI * 2);
          const gradient = ctx.createRadialGradient(
            node.x, node.y, 0,
            node.x, node.y, node.radius + 4
          );
          gradient.addColorStop(0, 'rgba(255, 215, 0, 0.6)');
          gradient.addColorStop(1, 'rgba(255, 215, 0, 0)');
          ctx.fillStyle = gradient;
          ctx.fill();
        }
        
        // Main node
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        
        if (node.active) {
          ctx.fillStyle = '#ffd700'; // Gold for active
          ctx.strokeStyle = '#ffed4e';
          ctx.lineWidth = 2;
        } else {
          ctx.fillStyle = '#374151'; // Gray for inactive
          ctx.strokeStyle = '#6b7280';
          ctx.lineWidth = 1;
        }
        
        ctx.fill();
        ctx.stroke();
        
        // Label layers
        if (node.y === Math.min(...nodes.filter(n => n.layer === node.layer).map(n => n.y))) {
          ctx.fillStyle = '#9ca3af';
          ctx.font = '12px Inter, sans-serif';
          ctx.textAlign = 'center';
          const labels = ['Input', 'Process', 'Analyze', 'Output'];
          ctx.fillText(labels[node.layer] || '', node.x, node.y - node.radius - 15);
        }
      });
      
      animationRef.current = requestAnimationFrame(animate);
    };

    resizeCanvas();
    createNodes();
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
    <div className={`w-full h-full relative ${className}`}>
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ background: 'transparent' }}
      />
      <div className="absolute bottom-4 left-4 text-sm text-foreground/60">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-analysis-green"></div>
            <span>Active Processing</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-muted"></div>
            <span>Idle</span>
          </div>
        </div>
      </div>
    </div>
  );
};