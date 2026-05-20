module.exports = {
  testEnvironment: 'jsdom',
  transform: {
    '^.+\\.jsx?$': 'babel-jest',
  },
  moduleNameMapper: {
    // Mock CSS imports
    '\\.(css|less|scss)$': 'identity-obj-proxy',
    // Mock elkjs (uses WASM/web workers, not compatible with jsdom)
    '^elkjs/lib/elk\\.bundled\\.js$': '<rootDir>/src/__mocks__/elkjs.js',
  },
  setupFilesAfterEnv: ['@testing-library/jest-dom'],
  testMatch: ['**/__tests__/**/*.test.{js,jsx}', '**/*.test.{js,jsx}'],
};
