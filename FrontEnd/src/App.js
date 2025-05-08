/*import React from 'react';
import './style.css';
import Navbar from './components/Navbar';
import HeroVideo from './components/HeroVideo';
import Intro from './components/Intro';
import './components/BasicCarousel.css'
import Categories from './components/Categories';
import TestButton from './components/test_button';
import FunctionalityCarousel from './components/FunctionalityCarousel';
import Footer from './components/Footer';
function App() {
  return (
    <>
      <Navbar />
      <HeroVideo />
      <Intro />
      <FunctionalityCarousel/>
      <Categories />
      <TestButton />
      <Footer/>
    </>
  );
}

export default App;*/

import React from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';

import Func from './components/Func';
import PredictPage from './components/PredictPage';
import Navbar from './components/Navbar';
import HeroVideo from './components/HeroVideo';
import Intro from './components/Intro';
import Categories from './components/Categories';
import FunctionalityCarousel from './components/FunctionalityCarousel';
import Footer from './components/Footer';

import './style.css';
import './components/BasicCarousel.css';

function AppWrapper() {
  const location = useLocation();
  const hideNavbar = location.pathname === '/predict';

  return (
    <>
      {!hideNavbar && <Navbar />}

      <Routes>
        <Route
          path="/"
          element={
            <>
              <HeroVideo />
              <Intro />
              <Categories />
              <FunctionalityCarousel />
              <Func />
              <Footer />
            </>
          }
        />
        <Route path="/predict" element={<PredictPage />} />
      </Routes>
    </>
  );
}

export default function App() {
  return (
    <Router>
      <AppWrapper />
    </Router>
  );
}

