import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react-swc';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, process.cwd(), '');

    return {
        plugins: [react()],
        server: {
            proxy: {
                '/connect': {
                    target: env.VITE_BACKEND_URL,   // dùng env
                    changeOrigin: true,
                },
            },
            allowedHosts: true,
            cors: true
        },
    };
});
