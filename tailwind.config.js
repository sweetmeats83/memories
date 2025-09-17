/** Tailwind CSS configuration for Memories app */
module.exports = {
  darkMode: 'class',
  content: [
    './templates/**/*.html',
    './static/js/**/*.js'
  ],
  safelist: [
    // Commonly toggled utilities in JS
    'hidden',
    'flex',
    'ring-1',
    'ring-2',
    'ring-black/10'
  ],
  theme: {
    extend: {
      colors: {
        accent: '#ff3bc1'
      }
    }
  },
  plugins: []
};
