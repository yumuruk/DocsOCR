/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#18202a"
      },
      boxShadow: {
        panel: "0 12px 32px rgba(17, 24, 39, 0.08)"
      }
    }
  },
  plugins: []
};

