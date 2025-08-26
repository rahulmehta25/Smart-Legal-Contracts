import { useState, useEffect } from 'react';

interface TypewriterTextProps {
  texts: string[];
  speed?: number;
  deleteSpeed?: number;
  pauseTime?: number;
  className?: string;
}

export const TypewriterText = ({ 
  texts, 
  speed = 100, 
  deleteSpeed = 50, 
  pauseTime = 2000,
  className = ""
}: TypewriterTextProps) => {
  const [currentTextIndex, setCurrentTextIndex] = useState(0);
  const [currentText, setCurrentText] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);
  const [isWaiting, setIsWaiting] = useState(false);

  useEffect(() => {
    const targetText = texts[currentTextIndex];
    let timeoutId: NodeJS.Timeout;

    if (isWaiting) {
      timeoutId = setTimeout(() => {
        setIsWaiting(false);
        setIsDeleting(true);
      }, pauseTime);
    } else if (isDeleting) {
      if (currentText === '') {
        setIsDeleting(false);
        setCurrentTextIndex((prev) => (prev + 1) % texts.length);
      } else {
        timeoutId = setTimeout(() => {
          setCurrentText(targetText.substring(0, currentText.length - 1));
        }, deleteSpeed);
      }
    } else {
      if (currentText === targetText) {
        setIsWaiting(true);
      } else {
        timeoutId = setTimeout(() => {
          setCurrentText(targetText.substring(0, currentText.length + 1));
        }, speed);
      }
    }

    return () => clearTimeout(timeoutId);
  }, [currentText, currentTextIndex, isDeleting, isWaiting, texts, speed, deleteSpeed, pauseTime]);

  return (
    <span className={`typewriter inline-block ${className}`}>
      {currentText}
      <span className="animate-pulse">|</span>
    </span>
  );
};