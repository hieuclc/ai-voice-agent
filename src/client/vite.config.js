import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react-swc';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, process.cwd(), '');

    return {
        plugins: [react()],
        server: {
            host: '0.0.0.0',      // ⭐ BẮT BUỘC
            port: 5173,           // ⭐ RÕ RÀNG
            strictPort: true,     // (khuyên dùng)
            proxy: {
                '/connect': {
                    target: env.VITE_API_BASE,
                    changeOrigin: true,
                },
            },
            allowedHosts: true,
            cors: true,
        },
    };
});
