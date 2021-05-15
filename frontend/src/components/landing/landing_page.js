import React from 'react';
import Button from '@material-ui/core/Button';
export default function LandingPage() {
    return(
        <div>
            <center>
                <h1 style = {{color: "#25DD63", 'font-size': '150px', 'margin-top': '10px'}}>FOCUS</h1>
                <a href = "/session">
                    <Button size = "large" style = {{'margin-top': '1px'}} variant="contained" color="primary">
                        Start Session
                    </Button>
                </a>
            </center>
        </div>
    );
}