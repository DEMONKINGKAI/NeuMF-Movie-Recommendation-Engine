import React, { useEffect, useRef, useState } from "react";
import "./App.css";
import anime from "https://esm.sh/animejs@3.2.2";
import GenreSelector from "./components/GenreSelector";
import StrictToggle from "./components/StrictToggle";
import Recommendations from "./components/Recommendations";
import { getGenres, getUsers, getRecommendations } from "./api";
import PromptSearch from "./components/PromptSearch";

export default function App() {
  const [genres, setGenres] = useState([]);
  const [users, setUsers] = useState([]);
  const [selectedGenre, setSelectedGenre] = useState("");
  const [selectedUser, setSelectedUser] = useState("");
  const [strict, setStrict] = useState(true);
  const [topK, setTopK] = useState(10);
  const [recs, setRecs] = useState([]);
  const [loading, setLoading] = useState(false);
  const titleRef = useRef(null);
  const cardRef = useRef(null);

  useEffect(() => {
    getGenres().then(setGenres);
    getUsers().then(us => {
      setUsers(us);
      if(us.length > 0) setSelectedUser(us[0]);
    });
  }, []);

  useEffect(() => {
    // Intro animations
    anime({ targets: titleRef.current, translateY: [16,0], opacity: [0,1], duration: 700, easing: 'easeOutQuad' });
    anime({ targets: cardRef.current, translateY: [18,0], opacity: [0,1], delay: 120, duration: 700, easing: 'easeOutQuad' });
  }, []);

  useEffect(() => {
    if (!selectedUser || !selectedGenre) return;
    setLoading(true);
    getRecommendations(selectedUser, selectedGenre, topK, strict).then((r) => {
      setRecs(r);
      setLoading(false);
    });
  }, [selectedUser, selectedGenre, topK, strict]);

  return (
    <div className="app-shell">
      <h1 ref={titleRef} className="title">NeuMF Movie Recommendations</h1>
      <div className="subtitle">Pick a user and genre to get personalized suggestions.</div>
      <div ref={cardRef} className="control-card">
        <div className="control">
          <span className="label">User</span>
          <select className="select" value={selectedUser} onChange={e => setSelectedUser(e.target.value)}>
            {users.map(u => (
              <option key={u} value={u}>{u}</option>
            ))}
          </select>
        </div>
        <div className="control">
          <span className="label">Genre</span>
          <GenreSelector genres={genres} value={selectedGenre} onChange={setSelectedGenre} />
        </div>
        <div className="control">
          <span className="label">Strict</span>
          <StrictToggle strict={strict} onChange={setStrict} />
        </div>
        <div className="control">
          <span className="label">Top K</span>
          <input className="input" style={{width:72}} type="number" min={1} max={50} value={topK} onChange={e=>setTopK(Number(e.target.value))} />
        </div>
      </div>
      <div className="control-card" style={{ marginTop: 16 }}>
        <PromptSearch
          userId={selectedUser}
          strict={strict}
          topK={topK}
          onResults={(r)=>setRecs(r)}
          onLoading={(v)=>setLoading(v)}
        />
      </div>
      <div className="results-card">
        {loading ? <div className="loader" /> : <Recommendations recommendations={recs} />}
      </div>
    </div>
  );
}