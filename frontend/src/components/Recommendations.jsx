import React, { useEffect, useRef } from "react";
import anime from "https://esm.sh/animejs@3.2.2";

export default function Recommendations({ recommendations }) {
  const listRef = useRef(null);

  useEffect(() => {
    if (!listRef.current) return;
    anime.remove(listRef.current.querySelectorAll('.tile'));
    anime({
      targets: listRef.current.querySelectorAll('.tile'),
      translateY: [10, 0],
      opacity: [0, 1],
      delay: anime.stagger(50),
      duration: 450,
      easing: 'easeOutQuad'
    });
  }, [recommendations]);

  if (!recommendations || !recommendations.length) return <div style={{padding:16, color:'#a6b0d6', textAlign:'center'}}>No recommendations yet.</div>;
  return (
    <ul ref={listRef} className="grid">
      {recommendations.map(rec => (
        <li key={rec.movie_id} className="tile">
          <div className="tile-title">{rec.title}</div>
          <div className="tile-meta">ID: {rec.movie_id}</div>
          <div className="tile-meta">Genres: {rec.genres.join(', ')}</div>
          <div className="tile-meta">Score: {rec.score.toFixed(3)}</div>
        </li>
      ))}
    </ul>
  );
}
