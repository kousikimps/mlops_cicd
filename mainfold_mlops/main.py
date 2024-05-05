from typing import Any
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import json

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger
from regression_model import __version__ as model_version
from regression_model.predict import make_prediction



print("model_version")
print("test started")




app = FastAPI(
    title="House Price Prediction App using API - CI CD Jenkins",
    description = "A Simple CI CD Demo",
    version='1.0'
)

origins=[
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)




from typing import Any, List, Optional

from pydantic import BaseModel
from regression_model.processing.validation import HouseDataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class MultipleHouseDataInputs(BaseModel):
    inputs: List[HouseDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "MSSubClass": 20,
                        "MSZoning": "RH",
                        "LotFrontage": 80.0,
                        "LotArea": 11622,
                        "Street": "Pave",
                        "Alley": None,
                        "LotShape": "Reg",
                        "LandContour": "Lvl",
                        "Utilities": "AllPub",
                        "LotConfig": "Inside",
                        "LandSlope": "Gtl",
                        "Neighborhood": "NAmes",
                        "Condition1": "Feedr",
                        "Condition2": "Norm",
                        "BldgType": "1Fam",
                        "HouseStyle": "1Story",
                        "OverallQual": 5,
                        "OverallCond": 6,
                        "YearBuilt": 1961,
                        "YearRemodAdd": 1961,
                        "RoofStyle": "Gable",
                        "RoofMatl": "CompShg",
                        "Exterior1st": "VinylSd",
                        "Exterior2nd": "VinylSd",
                        "MasVnrType": "None",
                        "MasVnrArea": 0.0,
                        "ExterQual": "TA",
                        "ExterCond": "TA",
                        "Foundation": "CBlock",
                        "BsmtQual": "TA",
                        "BsmtCond": "TA",
                        "BsmtExposure": "No",
                        "BsmtFinType1": "Rec",
                        "BsmtFinSF1": 468.0,
                        "BsmtFinType2": "LwQ",
                        "BsmtFinSF2": 144.0,
                        "BsmtUnfSF": 270.0,
                        "TotalBsmtSF": 882.0,
                        "Heating": "GasA",
                        "HeatingQC": "TA",
                        "CentralAir": "Y",
                        "Electrical": "SBrkr",
                        "FirstFlrSF": 896,
                        "SecondFlrSF": 0,
                        "LowQualFinSF": 0,
                        "GrLivArea": 896,
                        "BsmtFullBath": 0.0,
                        "BsmtHalfBath": 0.0,
                        "FullBath": 1,
                        "HalfBath": 0,
                        "BedroomAbvGr": 2,
                        "KitchenAbvGr": 1,
                        "KitchenQual": "TA",
                        "TotRmsAbvGrd": 5,
                        "Functional": "Typ",
                        "Fireplaces": 0,
                        "FireplaceQu": None,
                        "GarageType": "Attchd",
                        "GarageYrBlt": 1961.0,
                        "GarageFinish": "Unf",
                        "GarageCars": 1.0,
                        "GarageArea": 730.0,
                        "GarageQual": "TA",
                        "GarageCond": "TA",
                        "PavedDrive": "Y",
                        "WoodDeckSF": 140,
                        "OpenPorchSF": 0,
                        "EnclosedPorch": 0,
                        "ThreeSsnPortch": 0,
                        "ScreenPorch": 120,
                        "PoolArea": 0,
                        "PoolQC": None,
                        "Fence": "MnPrv",
                        "MiscFeature": None,
                        "MiscVal": 0,
                        "MoSold": 6,
                        "YrSold": 2010,
                        "SaleType": "WD",
                        "SaleCondition": "Normal",
                    }
                ]
            }
        }




@app.get("/")
def index(request: Request) -> Any:
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)




@app.post("/predict", response_model=PredictionResults, status_code=200)
async def predict(input_data: MultipleHouseDataInputs) -> Any:
    """
    Make house price predictions with the TID regression model
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    # Advanced: You can improve performance of your API by rewriting the
    # `make prediction` function to be async and using await here.
    logger.info(f"Making prediction on inputs: {input_data.inputs}")
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(f"Prediction results: {results.get('predictions')}")

    return results





if __name__ == "__main__":
    # Use this for debugging purposes only
    logger.warning("Running in development mode. Do not run like this in production.")
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8005)
