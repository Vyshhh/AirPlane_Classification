import React from 'react';
import vid from '../img/vid.mp4';

const HeroVideo = () => {
  return (
    <div className="hero-video">
      <video autoPlay muted loop>
        <source src={vid} type="video/mp4" />
        Your browser does not support the video tag.
      </video>
    </div>
  );
};

export default HeroVideo;