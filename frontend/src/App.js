import './App.css';
import axios from 'axios';
import React, { useState } from 'react';
function App() {
  const [text, settext] = useState('')
  const [res, setres] = useState('')
  const [click, setclick] = useState(false)
  return (
    <div className="App" >
      <div className='maincontainer'>
        <div className='innerContainer'>
          <h1>SPAM DETECTOR</h1>
          <div className='textareacontainer' >
            <textarea value={text} onChange={(event) => {
              settext(event.target.value)
            }} onClick={() => { setclick(true) }} />
          </div>
          <div className='buttonContainer'>
            <button onClick={() => {
              const response = axios({
                method: 'post',
                url: 'http://localhost:5000',
                data: {
                  text: text
                }
              }).catch(() => {
                console.log(console.error);
              })
              response.then((response) => {
                setres(response.data)
              })
            }}>Predict</button>
            <button onClick={() => { setres(''); settext('') }}>Clear</button>
          </div>
          <h1>{res}</h1>
        </div>
      </div>
    </div>
  );
}

export default App;
