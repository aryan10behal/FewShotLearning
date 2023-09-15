import './App.css';
import React, {useState} from 'react';
import axios from 'axios';
import {TextField, Grid, AppBar, Toolbar, Button,Input} from "@mui/material"

function App() {

  const [file, setFile] = useState();
  const [status, setStatus] = useState("");
  const [inputQuery, setInputQuery] = useState();
  const [response, setResponse] = useState();


  function handleChange(event) {setFile(event.target.files[0]);}

  function handleQueryChange(e) {setInputQuery(e.target.value)}

  function getQueryAnswer()
  {
    console.log(inputQuery);
    setResponse("(Loading your Results...)")
    const url = 'http://127.0.0.1:5000/query';
    axios.get(url,
      {
        params:
        {
          inputQuery: inputQuery,
        }
      }).then((response)=>{
        setResponse(response.data['result'])
      }).catch((e)=>{
        console.log(e)
      })
  }

  function getGPTModelAnswer()
  {
    const url = 'http://127.0.0.1:5000/queryGpt';
    setResponse("(Loading your Results...)")
    axios.get(url,
      {
        params:
        {
          inputQuery: inputQuery,
        }
      }).then((response)=>{
        setResponse(response.data['result'])
      }).catch((e)=>{
        console.log(e)
      })
  }
  
  function handleSubmit(event) {
    event.preventDefault();
    setStatus("Loading... Please wait")
    const url = 'http://127.0.0.1:5000/upload';
    const formData = new FormData();
    formData.append('file', file);
    formData.append('fileName', file.name);
    const config = {
      headers: {
        'content-type': 'multipart/form-data',
      },
    };
    axios.post(url, formData, config)
      .then((response) => {
        console.log(response);
        setStatus(response.data)
      })
      .catch((error) => {
        console.error("Error uploading file: ", error);
        setStatus("Error uploading file:" + error.message);
      });
  }

  return (
    <div className="App">
      <AppBar color='secondary' sx={{ backgroundColor: '#121d2c' }}>
          <Toolbar style={{ 
                    marginLeft  : 'auto',
                    marginRight : 'auto'}}
                    >
              <h1>Ask Me Anything</h1>
          </Toolbar>
      </AppBar>
      <Grid container rowSpacing={10} alignItems="center" justifyContent="center" border="1" margin={0}>
        <Grid item xs={6} md={8}>
          <form onSubmit={handleSubmit}>
            <h3>Upload your File Here</h3>
            <Input type="file" onChange={handleChange} id="icon-button-video"/>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            <Button style={{backgroundColor: "#8bb5e5",color: "#000000"}} type="submit">Upload</Button>
          </form>
          {<p>{status}</p>}
        </Grid>
        <Grid item xs={6} md={8}>
          <div> Enter Query: 
          <TextField
          fullWidth
          multiline={true}
          rows={3}
          width={100}
          value={inputQuery}
          onChange={(e)=>handleQueryChange(e)}
          ></TextField>
          <br></br>
          <br></br>
          <Button style={{backgroundColor: "#8bb5e5",color: "#000000"}} onClick={getQueryAnswer}>Query Free GPT</Button>
          &nbsp;&nbsp;&nbsp;&nbsp;
          <Button style={{backgroundColor: "#8bb5e5",color: "#000000"}} onClick={getGPTModelAnswer}>Query Msft GPT</Button>
          </div>
        </Grid>
        <Grid item xs={6} md={8}>
          <div>
            Result:
            <TextField
            fullWidth
          multiline={true}
          rows={5}
          width={100}
          value={response}
          ></TextField>
          </div>
        </Grid>
      </Grid>
    </div>
  );
}

export default App;