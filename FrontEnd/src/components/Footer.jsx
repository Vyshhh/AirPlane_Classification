// Footer.jsx
import React from 'react';
import './Footer.css';

export default function Footer() {
  return (
    <footer className="site-footer">
      <div className="footer-container">
        <p>&copy; {new Date().getFullYear()} Classifiers . All rights reserved.</p>
        
      </div>
    </footer>
  );
}
