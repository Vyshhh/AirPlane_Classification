import React from 'react';
import { Link } from 'react-router-dom';
import './func.css';

export default function Func() {
  return (
    <section id="thefunc" className="func">
      {/* ... your existing labels and functionalities ... */}
      <Link to="/predict" className="predict-button">
        Predict Now
      </Link>
    </section>
  );
}