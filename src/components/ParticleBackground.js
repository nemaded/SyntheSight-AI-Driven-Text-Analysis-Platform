// components/ParticleBackground.js
import React, { useEffect, useRef } from 'react';

const ParticleBackground = () => {
  const canvasRef = useRef(null);
  const mouseRef = useRef({ x: undefined, y: undefined, radius: 100 });
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Set canvas to full window size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    
    // Track mouse position
    const handleMouseMove = (event) => {
      mouseRef.current.x = event.x;
      mouseRef.current.y = event.y;
    };
    
    const handleMouseLeave = () => {
      mouseRef.current.x = undefined;
      mouseRef.current.y = undefined;
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseleave', handleMouseLeave);
    
    // Particle properties
    const particlesArray = [];
    const particleCount = Math.min(Math.floor(window.innerWidth * window.innerHeight / 10000), 100);
    const colors = ['#3f51b5', '#757de8', '#ff9800', '#ffc947', '#002984', '#c66900'];
    
    class Particle {
      constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = Math.random() * 5 + 1;
        this.baseSize = this.size;
        this.speedX = Math.random() * 1 - 0.5;
        this.speedY = Math.random() * 1 - 0.5;
        this.color = colors[Math.floor(Math.random() * colors.length)];
        this.opacity = Math.random() * 0.5 + 0.2;
        this.animationDelay = Math.random() * 2;
      }
      
      update() {
        // Normal movement
        this.x += this.speedX;
        this.y += this.speedY;
        
        // Interact with mouse
        const mouse = mouseRef.current;
        if (mouse.x !== undefined && mouse.y !== undefined) {
          const dx = mouse.x - this.x;
          const dy = mouse.y - this.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          if (distance < mouse.radius) {
            const forceDirectionX = dx / distance;
            const forceDirectionY = dy / distance;
            const force = (mouse.radius - distance) / mouse.radius;
            
            this.speedX -= forceDirectionX * force * 0.6;
            this.speedY -= forceDirectionY * force * 0.6;
            this.size = this.baseSize * (1 + force);
          } else {
            if (this.size > this.baseSize) {
              this.size -= 0.1;
            }
          }
        } else {
          if (this.size > this.baseSize) {
            this.size -= 0.1;
          }
        }
        
        // Bounce off edges with slight damping
        if (this.x > canvas.width || this.x < 0) {
          this.speedX = -this.speedX * 0.95;
          if (this.x > canvas.width) this.x = canvas.width;
          if (this.x < 0) this.x = 0;
        }
        if (this.y > canvas.height || this.y < 0) {
          this.speedY = -this.speedY * 0.95;
          if (this.y > canvas.height) this.y = canvas.height;
          if (this.y < 0) this.y = 0;
        }
        
        // Add some slight randomness to movement
        this.speedX += (Math.random() - 0.5) * 0.01;
        this.speedY += (Math.random() - 0.5) * 0.01;
        
        // Apply some drag
        this.speedX *= 0.99;
        this.speedY *= 0.99;
      }
      
      draw() {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fillStyle = this.color + Math.floor(this.opacity * 255).toString(16).padStart(2, '0');
        ctx.fill();
      }
    }
    
    // Create particles
    const init = () => {
      particlesArray.length = 0;
      for (let i = 0; i < particleCount; i++) {
        particlesArray.push(new Particle());
      }
    };
    
    // Animation loop
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw connection lines between particles that are close
      for (let i = 0; i < particlesArray.length; i++) {
        for (let j = i; j < particlesArray.length; j++) {
          const dx = particlesArray[i].x - particlesArray[j].x;
          const dy = particlesArray[i].y - particlesArray[j].y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          const maxDistance = 150;
          if (distance < maxDistance) {
            ctx.beginPath();
            ctx.strokeStyle = `rgba(140, 140, 200, ${0.8 - distance/maxDistance})`;
            ctx.lineWidth = 0.5;
            ctx.moveTo(particlesArray[i].x, particlesArray[i].y);
            ctx.lineTo(particlesArray[j].x, particlesArray[j].y);
            ctx.stroke();
          }
        }
      }
      
      // Update and draw particles
      for (let i = 0; i < particlesArray.length; i++) {
        particlesArray[i].update();
        particlesArray[i].draw();
      }
      
      requestAnimationFrame(animate);
    };
    
    init();
    animate();
    
    // Cleanup
    return () => {
      window.removeEventListener('resize', resizeCanvas);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, []);
  
  return (
    <canvas 
      ref={canvasRef} 
      className="particle-canvas"
    />
  );
};

export default ParticleBackground;