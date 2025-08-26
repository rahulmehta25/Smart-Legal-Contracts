export class ScreenReaderManager {
  private announceRegion: HTMLElement | null = null;

  constructor() {
    this.createAnnounceRegion();
  }

  private createAnnounceRegion(): void {
    this.announceRegion = document.createElement('div');
    this.announceRegion.setAttribute('aria-live', 'polite');
    this.announceRegion.setAttribute('aria-atomic', 'true');
    this.announceRegion.style.position = 'absolute';
    this.announceRegion.style.left = '-10000px';
    document.body.appendChild(this.announceRegion);
  }

  public announce(message: string, priority: 'polite' | 'assertive' = 'polite'): void {
    if (this.announceRegion) {
      this.announceRegion.setAttribute('aria-live', priority);
      this.announceRegion.textContent = message;
    }
  }

  public clear(): void {
    if (this.announceRegion) {
      this.announceRegion.textContent = '';
    }
  }
}