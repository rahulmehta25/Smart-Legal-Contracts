import { useEffect, useState } from 'react';
import { Appearance } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { getTheme } from '@utils/theme';
import { Theme } from '@types/index';

const THEME_STORAGE_KEY = 'app_theme_preference';

export const useTheme = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [isSystemTheme, setIsSystemTheme] = useState(true);
  const [theme, setTheme] = useState<Theme>(getTheme(false));

  useEffect(() => {
    loadThemePreference();
    
    const subscription = Appearance.addChangeListener(({ colorScheme }) => {
      if (isSystemTheme) {
        const isDark = colorScheme === 'dark';
        setIsDarkMode(isDark);
        setTheme(getTheme(isDark));
      }
    });

    return () => subscription?.remove();
  }, [isSystemTheme]);

  useEffect(() => {
    setTheme(getTheme(isDarkMode));
  }, [isDarkMode]);

  const loadThemePreference = async () => {
    try {
      const stored = await AsyncStorage.getItem(THEME_STORAGE_KEY);
      if (stored) {
        const preference = JSON.parse(stored);
        setIsSystemTheme(preference.isSystemTheme);
        
        if (preference.isSystemTheme) {
          const systemColorScheme = Appearance.getColorScheme();
          const isDark = systemColorScheme === 'dark';
          setIsDarkMode(isDark);
        } else {
          setIsDarkMode(preference.isDarkMode);
        }
      } else {
        // Default to system theme
        const systemColorScheme = Appearance.getColorScheme();
        const isDark = systemColorScheme === 'dark';
        setIsDarkMode(isDark);
      }
    } catch (error) {
      console.error('Error loading theme preference:', error);
    }
  };

  const toggleTheme = async () => {
    const newDarkMode = !isDarkMode;
    setIsDarkMode(newDarkMode);
    setIsSystemTheme(false);
    
    try {
      await AsyncStorage.setItem(
        THEME_STORAGE_KEY,
        JSON.stringify({
          isDarkMode: newDarkMode,
          isSystemTheme: false,
        })
      );
    } catch (error) {
      console.error('Error saving theme preference:', error);
    }
  };

  const useSystemTheme = async () => {
    setIsSystemTheme(true);
    const systemColorScheme = Appearance.getColorScheme();
    const isDark = systemColorScheme === 'dark';
    setIsDarkMode(isDark);
    
    try {
      await AsyncStorage.setItem(
        THEME_STORAGE_KEY,
        JSON.stringify({
          isDarkMode: isDark,
          isSystemTheme: true,
        })
      );
    } catch (error) {
      console.error('Error saving theme preference:', error);
    }
  };

  return {
    theme,
    isDarkMode,
    isSystemTheme,
    toggleTheme,
    useSystemTheme,
  };
};