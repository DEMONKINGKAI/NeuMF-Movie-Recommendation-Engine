import axios from 'axios';

const API = axios.create({ baseURL: 'http://localhost:8000' });

export const getGenres = async () => (await API.get('/genres')).data;
export const getUsers = async () => (await API.get('/users')).data;
export const getRecommendations = async (userId, genre, topK, strict) => (
    await API.get('/recommendations', {
        params: { user_id: userId, genre: genre, top_k: topK, strict: strict },
    })
).data;

export const getIntentRecommendations = async (q, userId, topK, strict, genreAlpha = 0.25, popAlpha = 0.15, embedAlpha = 0.20) => (
    await API.get('/intent_recommendations', {
        params: {
            q,
            user_id: userId,
            top_k: topK,
            strict,
            genre_alpha: genreAlpha,
            pop_alpha: popAlpha,
            embed_alpha: embedAlpha,
        },
    })
).data;