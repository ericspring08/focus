import Navbar from './components/navbar/navbar.js';
import LandingPage from './components/landing/landing_page.js';
import Session from './components/session/session.js';
import { Container } from '@material-ui/core';
import {Route, Link, BrowserRouter as Router} from 'react-router-dom';
function App() {
  return (
    <div>
      <Router>
        <Navbar></Navbar>
        <Route path = "/" exact component = {LandingPage}></Route>
        <Route path = "session" component = {Session}></Route>
      </Router>
    </div>
  );
}

export default App;
