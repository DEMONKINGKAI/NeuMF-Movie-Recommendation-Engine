import React from "react";
export default function StrictToggle({ strict, onChange }) {
  return (
    <label style={{ display:'flex', alignItems:'center', gap:8 }}>
      <input
        className="checkbox"
        type="checkbox"
        checked={strict}
        onChange={e => onChange(e.target.checked)}
      />
      Strict genre only
    </label>
  );
}
