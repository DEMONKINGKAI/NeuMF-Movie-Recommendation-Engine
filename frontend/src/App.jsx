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
  const [initializing, setInitializing] = useState(true);
  const titleRef = useRef(null);
  const cardRef = useRef(null);

  useEffect(() => {
    let mounted = true;
    setInitializing(true);
    
    Promise.all([
      getGenres()
        .then(data => {
          console.log('Genres loaded:', data);
          if (mounted) setGenres(data || []);
          return data;
        })
        .catch(err => {
          console.error('Failed to load genres:', err);
          if (mounted) setGenres([]);
          return [];
        }),
      getUsers()
        .then(us => {
          console.log('Users loaded:', us);
          if (mounted && us && us.length > 0) {
            setUsers(us);
            setSelectedUser(us[0]);
          } else {
            console.warn('No users found');
            if (mounted) setUsers([]);
          }
          return us;
        })
        .catch(err => {
          console.error('Failed to load users:', err);
          if (mounted) setUsers([]);
          return [];
        })
    ]).finally(() => {
      if (mounted) {
        setInitializing(false);
      }
    });
    
    return () => { mounted = false; };
  }, []);

  useEffect(() => {
    // Intro animations
    anime({ targets: titleRef.current, translateY: [16,0], opacity: [0,1], duration: 700, easing: 'easeOutQuad' });
    anime({ targets: cardRef.current, translateY: [18,0], opacity: [0,1], delay: 120, duration: 700, easing: 'easeOutQuad' });
  }, []);

  useEffect(() => {
    if (!selectedUser || !selectedGenre) return;
    setLoading(true);
    getRecommendations(selectedUser, selectedGenre, topK, strict)
      .then((r) => {
        console.log('Recommendations received:', r);
        setRecs(r || []);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to load recommendations:', err);
        setRecs([]);
        setLoading(false);
      });
  }, [selectedUser, selectedGenre, topK, strict]);

  if (initializing) {
    return (
      <div className="app-shell">
        <h1 ref={titleRef} className="title">NeuMF Movie Recommendations</h1>
        <div className="subtitle">Loading genres and users...</div>
        <div style={{ padding: '2rem', textAlign: 'center' }}>
          <div className="loader" />
          <p>Connecting to backend...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <h1 ref={titleRef} className="title">NeuMF Movie Recommendations</h1>
      <div className="subtitle">Pick a user and genre to get personalized suggestions.</div>
      <div ref={cardRef} className="control-card">
        <div className="control">
          <span className="label">User</span>
          <select className="select" value={selectedUser} onChange={e => setSelectedUser(e.target.value)} disabled={users.length === 0}>
            {users.length === 0 ? (
              <option>No users available</option>
            ) : (
              users.map(u => (
                <option key={u} value={u}>{u}</option>
              ))
            )}
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