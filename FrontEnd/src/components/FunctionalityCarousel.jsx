// FullWidthCarousel.jsx
/*import React, { useState, useEffect, useRef } from 'react';
import './BasicCarousel.css';

const images = [
  '/images_cou/uploaded_1746019736000.jpg',
  '/images_cou/Screenshot1.png',
  '/images_cou/Screenshot2.png',
];

export default function FullWidthCarousel() {
  const [current, setCurrent] = useState(0);
  const lastIndex = images.length - 1;
  const directionRef = useRef(1); // 1 = forward, -1 = backward

  // autoplay ping-pong every 5s
  useEffect(() => {
    const id = setInterval(() => {
      setCurrent(prev => {
        let next = prev + directionRef.current;
        if (next > lastIndex) {
          directionRef.current = -1;
          next = prev - 1;
        } else if (next < 0) {
          directionRef.current = 1;
          next = prev + 1;
        }
        return next;
      });
    }, 5000);
    return () => clearInterval(id);
  }, [lastIndex]);

  return (
    <div className="fullwidth-carousel">
      <div
        className="slides"
        style={{ transform: `translateX(-${current * 100}%)` }}
      >
        {images.map((src, i) => (
          <div className="slide" key={i}>
            <img
              src={src}
              alt={`Slide ${i + 1}`}
              className="slide-image"
            />
          </div>
        ))}
      </div>
    </div>
  );
}*/
/*
import React, { useState, useEffect, useRef } from 'react';
import './BasicCarousel.css';
import { Link } from 'react-router-dom';

// images array with captions
const slides = [
  { src: '/images_cou/image1.jpg', caption: 'Explore Military Aircraft' },
  { src: '/images_cou/image2.jpg', caption: 'Discover Civil Aircraft' },
  { src: '/images_cou/image3.jpg', caption: 'Analyze UAVs and Drones' },
];

export default function Func() {
  const [current, setCurrent] = useState(0);
  const lastIndex = slides.length - 1;
  const directionRef = useRef(1);

  useEffect(() => {
    const id = setInterval(() => {
      setCurrent(prev => {
        let next = prev + directionRef.current;
        if (next > lastIndex) {
          directionRef.current = -1;
          next = prev - 1;
        } else if (next < 0) {
          directionRef.current = 1;
          next = prev + 1;
        }
        return next;
      });
    }, 5000);
    return () => clearInterval(id);
  }, [lastIndex]);

  return (
    <section className="func">
      <div className="fullwidth-carousel">
        <div className="slides" style={{ transform: `translateX(-${current * 100}%)` }}>
          {slides.map(({ src, caption }, i) => (
            <div className="slide" key={i}>
              <img src={src} alt={`Slide ${i + 1}`} className="slide-image" />
              <div className="caption">{caption}</div>
            </div>
          ))}
        </div>
      </div>
     
    </section>
  );
}*/
import React from 'react';
import './BasicCarousel.css';

const slides = [
  { src: '/images_cou/image1.jpg', caption: 'Maintains a time-stamped log of all detected and classified aerial vehicles' },
  { src: '/images_cou/image2.jpg', caption: 'Integrates detection into a real-time pipeline capable of processing video feeds efficiently' },
  { src: '/images_cou/image3.jpg', caption: 'Efficiently processes and trains on a high-volume dataset' },
];

export default function FlashcardCarousel() {
  // Duplicate cards to allow seamless scrolling
  const allSlides = [...slides, ...slides];

  return (
    <div className="flashcard-strip">
      <div className="card-track">
        {allSlides.map((slide, index) => (
          <div
            className="card"
            key={index}
            style={{
              backgroundImage: `url(${slide.src})`,
            }}
          >
            {slide.caption}
          </div>
        ))}
      </div>
    </div>
  );
}
