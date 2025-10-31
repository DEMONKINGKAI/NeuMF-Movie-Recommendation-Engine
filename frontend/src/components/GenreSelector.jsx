import React from "react";
export default function GenreSelector({ genres, value, onChange }) {
  return (
    <select className="select" value={value} onChange={e => onChange(e.target.value)}>
      <option value="">Select Genre</option>
      {genres.map(g => (
        <option key={g} value={g}>{g}</option>
      ))}
    </select>
  );
}
