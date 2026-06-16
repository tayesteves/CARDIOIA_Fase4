/**
 * CardioIA — Configuração da API
 *
 * Durante o desenvolvimento:
 * - No navegador web, localhost funciona.
 * - No celular com Expo Go, use o IP local da máquina.
 */

const DEV_IP = "10.0.0.59";

const IS_WEB = typeof window !== "undefined";

export const API_URL =
  process.env.NODE_ENV === "production"
    ? "https://seu-backend-em-producao.com"
    : IS_WEB
      ? "http://localhost:5000"
      : `http://${DEV_IP}:5000`;

export const ENDPOINTS = {
  health: `${API_URL}/health`,
  classes: `${API_URL}/classes`,
  predict: `${API_URL}/predict`,
};
