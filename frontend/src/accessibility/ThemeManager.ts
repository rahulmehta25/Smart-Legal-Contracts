export class ThemeManager {
  private theme: 'light' | 'dark' | 'high-contrast' = 'light';

  constructor() {
    this.loadTheme();
  }

  private loadTheme(): void {
    const savedTheme = localStorage.getItem('accessibility-theme');
    if (savedTheme) {
      this.theme = savedTheme as 'light' | 'dark' | 'high-contrast';
      this.applyTheme();
    }
  }

  public setTheme(theme: 'light' | 'dark' | 'high-contrast'): void {
    this.theme = theme;
    this.applyTheme();
    localStorage.setItem('accessibility-theme', theme);
  }

  private applyTheme(): void {
    document.documentElement.setAttribute('data-theme', this.theme);
  }

  public getTheme(): 'light' | 'dark' | 'high-contrast' {
    return this.theme;
  }
}