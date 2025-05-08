import React from 'react';
import mili from '../img/mili.jpg';
import civil from '../img/civil.jpg';
import drone from '../img/drone.jpg';


const Categories = () => {
  return (
    <section id="categories" className="categories-section">
      <h2 className="categories-heading">Our Categories</h2>
      <div className="categories">
        <div className="category-box">
          <img src={mili} alt="Military Aircraft" />
          <p>Military Aircraft</p>
        </div>
        <div className="category-box">
          <img src={civil} alt="Civil Aircraft" />
          <p>Civil Aircraft</p>
        </div>
        <div className="category-box">
          <img src={drone} alt="UAV" />
          <p>Unmanned Aerial Vehicles</p>
        </div>
      </div>
      <h1 className="Our-Dataset">Our Dataset</h1>
      <div className="datainfo">
      Our dataset includes three balanced classes—Military Aircraft, Civil Aircraft, and Unmanned Aerial Vehicles,each containing 60,000 high-resolution images. These span everything from cutting-edge fighter jets and bombers to commercial airliners, private jets and a wide variety of hobbyist and industrial drones, providing comprehensive coverage for robust aviation classification models
      </div>
    </section>
  );
};

export default Categories;
