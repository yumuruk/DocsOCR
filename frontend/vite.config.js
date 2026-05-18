import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/documents": "http://127.0.0.1:8000",
      "/annotations": "http://127.0.0.1:8000",
      "/training": "http://127.0.0.1:8000",
      "/runs": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000"
    }
  }
});

