/** @type {import('tailwindcss').Config} */

// Colours resolve through the CSS custom properties defined in src/styles.css,
// so light and dark are a token swap and no component needs a `dark:` variant.
const token = (name) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        paper: token("paper"),
        surface: token("surface"),
        raised: token("raised"),
        line: token("line"),
        "line-strong": token("line-strong"),
        ink: token("ink"),
        muted: token("muted"),
        faint: token("faint"),
        action: token("action"),
        "action-ink": token("action-ink"),
        ring: token("ring"),
        danger: token("danger"),
        "danger-bg": token("danger-bg")
      },
      fontFamily: {
        sans: [
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI Variable",
          "Segoe UI",
          "sans-serif"
        ],
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "SF Mono",
          "Menlo",
          "Consolas",
          "monospace"
        ]
      },
      boxShadow: {
        proof: "0 1px 2px rgb(15 21 28 / 0.04), 0 8px 24px rgb(15 21 28 / 0.06)"
      }
    }
  },
  plugins: []
};
