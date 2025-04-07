import React from 'react';
import { Link } from 'react-router-dom';
import './Home.css';

const Home = () => {
  return (
    <div className="home">
      <h1>AI Fitness Assistant</h1>
      <p>Personalized workouts and diet plans just for you.</p>
      <div className="home-buttons">
        <Link to="/login"><button>Login</button></Link>
        <Link to="/signup"><button>Signup</button></Link>
      </div>
    </div>
  );
};

export default Home;
