import React, { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Sun, Moon, Monitor, Palette } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from '@/components/ui/dropdown-menu';

type Theme = 'light' | 'dark' | 'system';

interface ThemeToggleProps {
  className?: string;
  variant?: 'button' | 'dropdown' | 'switch';
}

export const ThemeToggle: React.FC<ThemeToggleProps> = ({
  className = '',
  variant = 'dropdown'
}) => {
  const [theme, setTheme] = useState<Theme>('system');
  const [mounted, setMounted] = useState(false);

  // Ensure component is mounted before rendering to avoid hydration mismatch
  useEffect(() => {
    setMounted(true);
    
    // Get saved theme from localStorage or default to system
    const savedTheme = localStorage.getItem('theme') as Theme | null;
    if (savedTheme) {
      setTheme(savedTheme);
      applyTheme(savedTheme);
    } else {
      applyTheme('system');
    }
  }, []);

  const applyTheme = (newTheme: Theme) => {
    const root = document.documentElement;
    
    if (newTheme === 'dark') {
      root.classList.add('dark');
    } else if (newTheme === 'light') {
      root.classList.remove('dark');
    } else {
      // System theme
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      if (mediaQuery.matches) {
        root.classList.add('dark');
      } else {
        root.classList.remove('dark');
      }
    }
  };

  const setThemeWithPersistence = (newTheme: Theme) => {
    setTheme(newTheme);
    localStorage.setItem('theme', newTheme);
    applyTheme(newTheme);
  };

  // Listen for system theme changes
  useEffect(() => {
    if (theme === 'system') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      
      const handleChange = (e: MediaQueryListEvent) => {
        if (theme === 'system') {
          applyTheme('system');
        }
      };

      mediaQuery.addEventListener('change', handleChange);
      return () => mediaQuery.removeEventListener('change', handleChange);
    }
  }, [theme]);

  if (!mounted) {
    return null; // Avoid hydration mismatch
  }

  const getThemeIcon = (themeType: Theme) => {
    switch (themeType) {
      case 'light':
        return <Sun className="w-4 h-4" />;
      case 'dark':
        return <Moon className="w-4 h-4" />;
      case 'system':
        return <Monitor className="w-4 h-4" />;
      default:
        return <Palette className="w-4 h-4" />;
    }
  };

  const getCurrentIcon = () => {
    if (theme === 'system') {
      const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      return isDark ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />;
    }
    return getThemeIcon(theme);
  };

  // Simple button variant
  if (variant === 'button') {
    const toggleTheme = () => {
      const newTheme = theme === 'light' ? 'dark' : 'light';
      setThemeWithPersistence(newTheme);
    };

    return (
      <Button
        variant="ghost"
        size="sm"
        onClick={toggleTheme}
        className={`focus-ring ${className}`}
        aria-label="Toggle theme"
      >
        {getCurrentIcon()}
      </Button>
    );
  }

  // Switch variant
  if (variant === 'switch') {
    return (
      <div className={`flex items-center space-x-2 ${className}`}>
        <Sun className="w-4 h-4 text-muted-foreground" />
        <button
          onClick={() => setThemeWithPersistence(theme === 'light' ? 'dark' : 'light')}
          className={`
            relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2
            ${theme === 'dark' ? 'bg-primary' : 'bg-muted'}
          `}
          role="switch"
          aria-checked={theme === 'dark'}
          aria-label="Toggle dark mode"
        >
          <span
            className={`
              inline-block h-4 w-4 transform rounded-full bg-background transition-transform
              ${theme === 'dark' ? 'translate-x-6' : 'translate-x-1'}
            `}
          />
        </button>
        <Moon className="w-4 h-4 text-muted-foreground" />
      </div>
    );
  }

  // Dropdown variant (default)
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className={`focus-ring ${className}`}
          aria-label="Theme settings"
        >
          {getCurrentIcon()}
          <span className="sr-only">Toggle theme</span>
        </Button>
      </DropdownMenuTrigger>
      
      <DropdownMenuContent align="end" className="w-48">
        <DropdownMenuLabel>Theme Settings</DropdownMenuLabel>
        <DropdownMenuSeparator />
        
        <DropdownMenuItem
          onClick={() => setThemeWithPersistence('light')}
          className={`cursor-pointer ${theme === 'light' ? 'bg-accent' : ''}`}
        >
          <Sun className="w-4 h-4 mr-2" />
          <span>Light Mode</span>
          {theme === 'light' && (
            <div className="ml-auto w-2 h-2 bg-primary rounded-full" />
          )}
        </DropdownMenuItem>
        
        <DropdownMenuItem
          onClick={() => setThemeWithPersistence('dark')}
          className={`cursor-pointer ${theme === 'dark' ? 'bg-accent' : ''}`}
        >
          <Moon className="w-4 h-4 mr-2" />
          <span>Dark Mode</span>
          {theme === 'dark' && (
            <div className="ml-auto w-2 h-2 bg-primary rounded-full" />
          )}
        </DropdownMenuItem>
        
        <DropdownMenuItem
          onClick={() => setThemeWithPersistence('system')}
          className={`cursor-pointer ${theme === 'system' ? 'bg-accent' : ''}`}
        >
          <Monitor className="w-4 h-4 mr-2" />
          <span>System</span>
          {theme === 'system' && (
            <div className="ml-auto w-2 h-2 bg-primary rounded-full" />
          )}
        </DropdownMenuItem>
        
        <DropdownMenuSeparator />
        
        <div className="px-2 py-1 text-xs text-muted-foreground">
          Current: {theme === 'system' 
            ? `System (${window.matchMedia('(prefers-color-scheme: dark)').matches ? 'Dark' : 'Light'})`
            : theme.charAt(0).toUpperCase() + theme.slice(1)
          }
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

// Theme provider hook for easier theme management
export const useTheme = () => {
  const [theme, setTheme] = useState<Theme>('system');

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') as Theme | null;
    if (savedTheme) {
      setTheme(savedTheme);
    }
  }, []);

  const setThemeWithPersistence = (newTheme: Theme) => {
    setTheme(newTheme);
    localStorage.setItem('theme', newTheme);
    
    const root = document.documentElement;
    if (newTheme === 'dark') {
      root.classList.add('dark');
    } else if (newTheme === 'light') {
      root.classList.remove('dark');
    } else {
      // System theme
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      if (mediaQuery.matches) {
        root.classList.add('dark');
      } else {
        root.classList.remove('dark');
      }
    }
  };

  return {
    theme,
    setTheme: setThemeWithPersistence
  };
};

export default ThemeToggle;