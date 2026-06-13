async function predict(){


let data = {


funding_rounds:
Number(document.getElementById("funding_rounds").value),


founder_experience_years:
Number(document.getElementById("founder_experience_years").value),


team_size:
Number(document.getElementById("team_size").value),


market_size_billion:
Number(document.getElementById("market_size_billion").value),


product_traction_users:
Number(document.getElementById("product_traction_users").value),


burn_rate_million:
Number(document.getElementById("burn_rate_million").value),


revenue_million:
Number(document.getElementById("revenue_million").value),


investor_type:
Number(document.getElementById("investor_type").value),


sector:
Number(document.getElementById("sector").value),


founder_background:
Number(document.getElementById("founder_background").value)


};




let response = await fetch(

"http://127.0.0.1:8000/predict",

{

method:"POST",

headers:{

"Content-Type":"application/json"

},

body:JSON.stringify(data)

}

);



let result = await response.json();



document.getElementById("result").innerHTML =

"Prediction: "
+
result.prediction;



document.getElementById("confidence").innerHTML =

"Confidence: "
+
result.confidence
+
"%";



}