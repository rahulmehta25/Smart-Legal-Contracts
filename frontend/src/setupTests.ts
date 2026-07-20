/**
 * Jest test setup configuration for arbitration detection frontend.
 * 
 * This file is automatically run before each test file and sets up:
 * - Custom matchers from @testing-library/jest-dom
 * - Global test utilities and mocks
 * - Environment configuration for testing
 */

import '@testing-library/jest-dom'

// Mock Next.js router
jest.mock('next/router', () => ({
  useRouter() {
    return {
      route: '/',
      pathname: '/',
      query: {},
      asPath: '/',
      push: jest.fn(),
      pop: jest.fn(),
      reload: jest.fn(),
      back: jest.fn(),
      prefetch: jest.fn().mockResolvedValue(undefined),
      beforePopState: jest.fn(),
      events: {
        on: jest.fn(),
        off: jest.fn(),
        emit: jest.fn(),
      },
      isFallback: false,
    }
  },
}))

// Mock Next.js Image component
jest.mock('next/image', () => ({
  __esModule: true,
  default: (props: any) => {
    // eslint-disable-next-line jsx-a11y/alt-text, @next/next/no-img-element
    return <img {...props} />
  },
}))

// Mock file uploads
global.File = class MockFile {
  name: string
  size: number
  type: string
  lastModified: number
  
  constructor(chunks: any[], filename: string, options: any = {}) {
    this.name = filename
    this.size = chunks.reduce((acc, chunk) => acc + chunk.length, 0)
    this.type = options.type || ''
    this.lastModified = Date.now()
  }
} as any

global.FileReader = class MockFileReader {
  result: any = null
  error: any = null
  readyState: number = 0
  onload: any = null
  onerror: any = null
  onloadend: any = null
  
  readAsText(file: any) {
    this.result = 'Mock file content'
    this.readyState = 2
    if (this.onload) this.onload({ target: this })
    if (this.onloadend) this.onloadend({ target: this })
  }
  
  readAsDataURL(file: any) {
    this.result = 'data:text/plain;base64,TW9jayBmaWxlIGNvbnRlbnQ='
    this.readyState = 2
    if (this.onload) this.onload({ target: this })
    if (this.onloadend) this.onloadend({ target: this })
  }
} as any

// Mock IntersectionObserver
global.IntersectionObserver = class MockIntersectionObserver {
  observe = jest.fn()
  disconnect = jest.fn()
  unobserve = jest.fn()
} as any

// Mock ResizeObserver
const ResizeObserver = require('resize-observer-polyfill')
global.ResizeObserver = ResizeObserver

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // deprecated
    removeListener: jest.fn(), // deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
})

// Mock scrollTo
Object.defineProperty(window, 'scrollTo', {
  writable: true,
  value: jest.fn(),
})

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
}
Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
})

// Mock sessionStorage
const sessionStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
}
Object.defineProperty(window, 'sessionStorage', {
  value: sessionStorageMock,
})

// Mock console methods to reduce noise in tests
const originalError = console.error
const originalWarn = console.warn

console.error = (...args: any[]) => {
  // Suppress specific warnings/errors that are expected in tests
  const message = args[0]
  if (
    typeof message === 'string' &&
    (message.includes('Warning: ReactDOM.render is deprecated') ||
     message.includes('Warning: componentWillReceiveProps') ||
     message.includes('Warning: componentWillMount'))
  ) {
    return
  }
  originalError(...args)
}

console.warn = (...args: any[]) => {
  const message = args[0]
  if (
    typeof message === 'string' &&
    message.includes('Warning: ')
  ) {
    return
  }
  originalWarn(...args)
}

// Global test utilities
global.testUtils = {
  // Mock API responses
  mockApiResponse: (data: any, status = 200) => ({
    data,
    status,
    statusText: 'OK',
    headers: {},
    config: {},
  }),
  
  // Mock file upload
  createMockFile: (name: string, content: string, type = 'text/plain') => {
    return new File([content], name, { type })
  },
  
  // Wait for async operations
  waitFor: (callback: () => boolean, timeout = 1000) => {
    return new Promise((resolve, reject) => {
      const startTime = Date.now()
      const checkCondition = () => {
        if (callback()) {
          resolve(true)
        } else if (Date.now() - startTime > timeout) {
          reject(new Error('Timeout waiting for condition'))
        } else {
          setTimeout(checkCondition, 10)
        }
      }
      checkCondition()
    })
  },
}

// Clean up after each test
afterEach(() => {
  // Clear all mocks
  jest.clearAllMocks()
  
  // Clear localStorage
  localStorageMock.getItem.mockClear()
  localStorageMock.setItem.mockClear()
  localStorageMock.removeItem.mockClear()
  localStorageMock.clear.mockClear()
  
  // Clear sessionStorage
  sessionStorageMock.getItem.mockClear()
  sessionStorageMock.setItem.mockClear()
  sessionStorageMock.removeItem.mockClear()
  sessionStorageMock.clear.mockClear()
})

// Global error handler for unhandled promises
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason)
})

// Type declarations for global test utilities
declare global {
  var testUtils: {
    mockApiResponse: (data: any, status?: number) => any
    createMockFile: (name: string, content: string, type?: string) => File
    waitFor: (callback: () => boolean, timeout?: number) => Promise<boolean>
  }
}