import pandas as pd
import torch
import torch.nn as nn


class mySimpleNN(nn.Module):

    def __init__(self, num_features):

        super().__init__()

        self.model = nn.Sequential(

            # Hidden Layer
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

            # Output Layer
            nn.Linear(
                64,
                3
            )
        )

    def forward(self, X):

        return self.model(X)
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model = mySimpleNN(
    10       # your 10 input features
)
model.load_state_dict(
    torch.load(
        "startup_prediction_model (1).pth",
        map_location=device
    )
)
model.to(device)

model.eval()



# ==============================
# 5. Give New Startup Data
# ==============================

# Order must be exactly the same as training

input_data = [

    5,          # funding_rounds

    8,          # founder_experience_years

    40,         # team_size

    25.5,       # market_size_billion

    120000,     # product_traction_users

    4.5,        # burn_rate_million

    8.2,        # revenue_million

    1,          # investor_type (encoded)

    2,          # sector (encoded)

    0           # founder_background (encoded)

]



# ==============================
# 6. Convert Input To Tensor
# ==============================

input_tensor = torch.tensor(
    input_data,
    dtype=torch.float32
)


# Add batch dimension
# [10] --> [1,10]

input_tensor = input_tensor.unsqueeze(0)


input_tensor = input_tensor.to(device)



# ==============================
# 7. Prediction
# ==============================

with torch.no_grad():

    output = model(input_tensor)


    prediction = torch.argmax(
        output,
        dim=1
    )



# ==============================
# 8. Convert Prediction To Label
# ==============================

classes = {

    0: "Failure",

    1: "Acquisition",

    2: "IPO"

}


result = classes[
    prediction.item()
]


print("Startup Prediction:", result)