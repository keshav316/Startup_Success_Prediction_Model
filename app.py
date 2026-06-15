from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle


# ==========================
# Load Scaler
# ==========================

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)



# ==========================
# FastAPI App
# ==========================

app = FastAPI()



app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)



# ==========================
# Home Route
# ==========================

@app.get("/")
def home():

    return {
        "message": "Startup Prediction API is running"
    }




# ==========================
# Model Architecture
# ==========================

class mySimpleNN(nn.Module):

    def __init__(self, num_features):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(
                num_features,
                64
            ),

            nn.BatchNorm1d(
                64,
                momentum=0.22224623206758784,
                eps=9.98053794949822e-06
            ),

            nn.ReLU(),

            nn.Dropout(
                0.16985411441075865
            ),

            nn.Linear(
                64,
                3
            )
        )


    def forward(self, x):

        return self.model(x)




# ==========================
# Load Model
# ==========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



model = mySimpleNN(
    num_features=10
)



model.load_state_dict(

    torch.load(
        "startup_prediction_model (1).pth",
        map_location=device
    )

)


model.to(device)

model.eval()




# ==========================
# Input Schema
# ==========================

class StartupData(BaseModel):

    funding_rounds: int

    founder_experience_years: int

    team_size: int

    market_size_billion: float

    product_traction_users: int

    burn_rate_million: float

    revenue_million: float

    investor_type: int

    sector: int

    founder_background: int





# ==========================
# Prediction API
# ==========================

@app.post("/predict")
def predict(data: StartupData):


    input_data = [

        data.funding_rounds,

        data.founder_experience_years,

        data.team_size,

        data.market_size_billion,

        data.product_traction_users,

        data.burn_rate_million,

        data.revenue_million,

        data.investor_type,

        data.sector,

        data.founder_background

    ]



    # Apply same scaling used during training
    input_data = scaler.transform(
        [input_data]
    )



    # Convert numpy array to tensor
    input_tensor = torch.tensor(
        input_data,
        dtype=torch.float32
    )



    input_tensor = input_tensor.to(device)



    print("Input tensor shape:", input_tensor.shape)



    with torch.no_grad():

        output = model(input_tensor)



        probability = F.softmax(
            output,
            dim=1
        )



        confidence, prediction = torch.max(
            probability,
            dim=1
        )



    classes = {

        0: "Failure",

        1: "Acquisition",

        2: "IPO"

    }



    return {

        "prediction": classes[prediction.item()],

        "confidence": round(
            confidence.item()*100,
            2
        )

    }