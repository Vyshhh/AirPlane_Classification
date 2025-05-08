import React from 'react';

const Navbar = () => {
  return (
    <nav>
      <div className="logo">Aerial Vehicle Detector</div>
      <ul>
        <li><a href="#">Home</a></li>
        <li><a href="#intro">About</a></li>
        <li><a href="#categories">Categories</a></li>
        <li><a href="#thefunc">Test</a></li>
      </ul>
    </nav>
  );
};

export default Navbar;