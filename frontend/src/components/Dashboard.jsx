import React from 'react';
import './Dashboard.css';

const Dashboard = () => {
  return (
    <div className="dashboard">
      <h2>Your Fitness Dashboard</h2>
      <p>Here’s your personalized workout and diet recommendation.</p>

      <div className="dashboard-cards">
        <div className="card">Workout Plan</div>
        <div className="card">Diet Plan</div>
        <div className="card">Progress Tracker</div>
      </div>
    </div>
  );
};

export default Dashboard;
