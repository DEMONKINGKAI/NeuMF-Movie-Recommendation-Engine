import axios from 'axios';

const API = axios.create({ 
  baseURL: 'http://localhost:8000',
  timeout: 30000, // 30 second timeout
});

// Add request interceptor for logging
API.interceptors.request.use(
  config => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`, config.params);
    return config;
  },
  error => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for logging
API.interceptors.response.use(
  response => {
    console.log(`API Response: ${response.config.url}`, response.status);
    return response;
  },
  error => {
    console.error('API Response Error:', error.response?.status, error.message);
    if (error.code === 'ECONNABORTED') {
      console.error('Request timeout - backend may be slow or unresponsive');
    }
    return Promise.reject(error);
  }
);

export const getGenres = async () => {
  try {
    const response = await API.get('/genres');
    return response.data;
  } catch (error) {
    console.error('getGenres error:', error);
    throw error;
  }
};

export const getUsers = async () => {
  try {
    const response = await API.get('/users');
    return response.data;
  } catch (error) {
    console.error('getUsers error:', error);
    throw error;
  }
};

export const getRecommendations = async (userId, genre, topK, strict) => {
  try {
    const response = await API.get('/recommendations', {
      params: { user_id: userId, genre: genre, top_k: topK, strict: strict },
    });
    return response.data;
  } catch (error) {
    console.error('getRecommendations error:', error);
    throw error;
  }
};

export const getIntentRecommendations = async (q, userId, topK, strict, genreAlpha = 0.25, popAlpha = 0.15, embedAlpha = 0.20) => {
  try {
    const response = await API.get('/intent_recommendations', {
      params: {
        q,
        user_id: userId,
        top_k: topK,
        strict,
        genre_alpha: genreAlpha,
        pop_alpha: popAlpha,
        embed_alpha: embedAlpha,
      },
    });
    return response.data;
  } catch (error) {
    console.error('getIntentRecommendations error:', error);
    throw error;
  }
};