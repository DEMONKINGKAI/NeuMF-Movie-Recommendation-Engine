import React, { useState } from 'react';
import { getIntentRecommendations } from '../api';

export default function PromptSearch({ userId, strict, topK, onResults, onLoading }) {
  const [q, setQ] = useState('');
  const [genreAlpha, setGenreAlpha] = useState(0.25);
  const [popAlpha, setPopAlpha] = useState(0.15);
  const [embedAlpha, setEmbedAlpha] = useState(0.20);

  const submit = async (e) => {
    e.preventDefault();
    if (!userId || !q.trim()) return;
    onLoading?.(true);
    try {
      const recs = await getIntentRecommendations(q, userId, topK, strict, genreAlpha, popAlpha, embedAlpha);
      onResults(recs);
    } finally {
      onLoading?.(false);
    }
  };

  return (
    <form onSubmit={submit} style={{ display: 'flex', flexDirection: 'column', gap: 6, width: '100%' }}>
      <div style={{ display: 'flex', gap: 8, width: '100%' }}>
        <input
          className="input"
          style={{ flex: 1, minWidth: 0 }}
          type="text"
          placeholder="Describe a movie (e.g., 'exciting, thrilling, awesome')"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
        <button className="button" type="submit" disabled={!q.trim()}>Recommend</button>
      </div>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <div className="control" style={{ margin: 0 }}>
          <span className="label">Genre α</span>
          <input className="input" style={{ width: 72 }} type="number" step="0.05" value={genreAlpha} onChange={e=>setGenreAlpha(parseFloat(e.target.value))} />
        </div>
        <div className="control" style={{ margin: 0 }}>
          <span className="label">Popularity α</span>
          <input className="input" style={{ width: 72 }} type="number" step="0.05" value={popAlpha} onChange={e=>setPopAlpha(parseFloat(e.target.value))} />
        </div>
        <div className="control" style={{ margin: 0 }}>
          <span className="label">Embedding α</span>
          <input className="input" style={{ width: 72 }} type="number" step="0.05" value={embedAlpha} onChange={e=>setEmbedAlpha(parseFloat(e.target.value))} />
        </div>
      </div>
    </form>
  );
}


