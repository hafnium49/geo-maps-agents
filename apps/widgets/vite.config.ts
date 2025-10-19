import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: path.resolve(__dirname, '../mcp_server/assets'),
    emptyOutDir: true,
    sourcemap: true,
    lib: {
      entry: path.resolve(__dirname, 'src/bootstrap.ts'),
      name: 'GeoMapsWidgets',
      formats: ['es'],
      fileName: () => 'widgets.js'
    },
    rollupOptions: {
      external: [],
    },
  },
});
