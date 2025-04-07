// Firebase App (the core Firebase SDK)
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.0/firebase-app.js";
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/10.7.0/firebase-auth.js";

// Your Firebase config
const firebaseConfig = {
    apiKey: "AIzaSyA3SVXU8JC1sSMOWXpckpUUza5zc-ZoIBc",
    authDomain: "project-new-cd7e2.firebaseapp.com",
    projectId: "project-new-cd7e2",
    storageBucket: "project-new-cd7e2.firebasestorage.app",
    messagingSenderId: "500836607265",
    appId: "1:500836607265:web:c68c0d9024b50fe6cbca60"
  };

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// Login Function
window.login = function () {
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;

  signInWithEmailAndPassword(auth, email, password)
    .then((userCredential) => {
      alert("Login Successful!");
      window.location.href = "index.html";
    })
    .catch((error) => {
      alert(error.message);
    });
};

// Signup Function
window.signup = function () {
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;

  createUserWithEmailAndPassword(auth, email, password)
    .then((userCredential) => {
      alert("Signup Successful!");
      window.location.href = "index.html";
    })
    .catch((error) => {
      alert(error.message);
    });
};
