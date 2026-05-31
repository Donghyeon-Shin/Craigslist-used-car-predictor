import pandas as pd

# Read the preprocessed CSV file
df = pd.read_csv('./Data/vehicles_10000.csv')

VIN_List = df["VIN"].tolist()
Model_List = df["model"].tolist()

df.shape

# The VIN API is the free API provided by the NHTSA (National Highway Traffic Safety Administration) of the US.
# Requests are split into batches of 50 records, and each result is saved to the `vin_api_result.csv` file.
#
# The VIN API provides the following information for analyzing the original retail (MSRP) price:
# - Make: Determines the brand premium (Brand Tax).
# - Model: Fixes the vehicle segment (compact, mid-size, full-size, etc.).
# - Series: Distinguishes the weight/size class among pickup trucks and full-size vans.
# - Trim: Reflects the added price of option packages by trim level.
# - FuelTypePrimary: Price differs depending on the fuel type.
# - DriveType: 4WD options are more expensive than 2WD.
# - EngineCylinders & DisplacementL: Options that may be missing from the trim information.
# - TransmissionStyle: PDK or manual transmissions add a price premium for enthusiast sports cars compared to automatics.
# - Doors: For trucks, the price differs depending on whether it has 2 or 4 doors.
#
# In addition, because a seller may have entered a manipulated VIN in the original data, we add a feature that checks whether the two model columns match.
# Note that in the original data the seller may have included the series in the model, and spaces/hyphens and letter case may all differ, so we must preprocess before comparing.
#
# - IsFraud: Determines whether the VIN has been manipulated.
# - Log: The model name from the original data when IsFraud is True.

import re

def check_vin_fraud(origin_data_model, vin_api_result_model):
    if origin_data_model == None or vin_api_result_model == None:
        return True
    
    # normalize text (lowercase, remove space, remove hyphen)
    def normalize_text(text):
        return re.sub(r'[^a-z0-9]', '', str(text).lower())

    normalized_origin_model = normalize_text(origin_data_model)
    normalized_vin_model = normalize_text(vin_api_result_model)
    
    # check if normalized_origin_model is in normalized_vin_model
    if (normalized_origin_model in normalized_vin_model) or (normalized_vin_model in normalized_origin_model):
        return False
    else:
        return True

import requests
import time
from tqdm import tqdm

# NHTSA vPIC(Vehicle Product Information Catalog) API
# Don't Need API KEY
API_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVINValuesBatch/"

# We need to get these features from VIN API
FEATURES = ["VIN", "ModelYear", "Make", "Model", "Series", "FuelTypePrimary", "Trim", "DriveType", "EngineCylinders", "DisplacementL", "TransmissionStyle", "Doors", "IsFraud", "Log"]

def fetch_to_csv(VIN_List, model_List, output_file = "vin_api_result.csv"):
    results = []

    # split 50 batch for request
    for i in tqdm(range(0, len(VIN_List), 50)):
        vin_chuck = VIN_List[i:i+50]
        model_chuck = model_List[i:i+50]

        api_results = []

        # fill null value for each feature (init states)
        for v in vin_chuck:
            place_holder = {feat: None for feat in FEATURES}
            place_holder["VIN"] = v
            api_results.append(place_holder)

        # request to VIN API
        try :
            payload = {"format": "json", "data": ";".join(vin_chuck)}
            response = requests.post(API_URL, data=payload, timeout=20)
            response.raise_for_status()

            api_data = response.json().get("Results", [])
            
            # fill api_result with api_data
            for res in api_data:
                target_vin = res.get("VIN")
                for p in api_results:
                    if p["VIN"] == target_vin:
                        for k, v in res.items():
                            if k in FEATURES:
                                p[k] = v

            # Check fraud
            for idx, p in enumerate(api_results):
                Isfraud = check_vin_fraud(model_chuck[idx], p.get("Model"))
                if Isfraud:
                    p["IsFraud"] = True
                    p["Log"] = model_chuck[idx]
                else:
                    p["IsFraud"] = False
                    p["Log"] = None
        
        except Exception as e:
            print(f"\n[Error] Problem occurred while processing batch {i} (replaced with null): {e}")

        results.extend(api_results)
        time.sleep(1) # prevent rate limit

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"VIN API results saved: {output_file}")

fetch_to_csv(VIN_List, Model_List, "./Data/vin_api_result.csv")

import pandas as pd

# Read VIN api Result file
vin_api_result = pd.read_csv("Data/vin_api_result.csv")
print(f"Number of Make categories: {vin_api_result['Make'].nunique()}")
print(f"Number of Model categories: {vin_api_result['Model'].nunique()}")
print(f"Number of Series categories: {vin_api_result['Series'].nunique()}")
vin_api_result.isnull().sum()

# Based on the VIN API results, we use an LLM to create the estimated original retail (MSRP) price feature.
# Note: when describing each feature to the LLM, we must NOT use specific numbers, to avoid anchoring bias.
# Also, we must NOT include the used price in the prompt, since the LLM could infer the answer from it.
# Finally, we instruct the LLM to output a Null result instead of guessing uncertain information, to minimize hallucination.
#
# + The result is parsed and returned as a Dictionary object.

# Model Prompt
msrp_prompt_template = """
[System Role]
You are an expert automotive pricing actuary and US used car market specialist.
Your exact task is to retrieve or strictly estimate the "Original MSRP (Manufacturer's Suggested Retail Price)" in USD for a vehicle based on its VIN-decoded features. 

[Pricing Logic & Feature Importance]
Do NOT use simple addition. You must retrieve the historical MSRP based on your internal knowledge of the specific Make, Model, and Year, while strictly accounting for the following premium factors:
1. Make, Model & Series: Determine the base MSRP for the exact vehicle segment and weight class (e.g., F-150 vs F-250 base prices vary significantly).
2. Trim: This is the most critical factor. Adjust the MSRP to reflect the specific trim level's standard equipment and luxury packages.
3. FuelTypePrimary: Apply the historical market premium for Diesel or Hybrid powertrains compared to standard Gasoline for that specific model.
4. DriveType: Ensure the MSRP reflects the cost of a 4WD/AWD drivetrain if equipped, as it carries a premium over standard 2WD.
5. EngineCylinders & DisplacementL: Account for the engine upgrade costs (e.g., the historical price difference between a base V6 and an optional V8).
6. TransmissionStyle: Consider premiums for specialized transmissions (e.g., PDK) or enthusiast Manual setups in sports cars.
7. Doors: For trucks, strictly use this to identify the Cab configuration (e.g., 2-door Regular Cab vs. 4-door Crew Cab) and price accordingly.

[Strict Constraints - NO HALLUCINATION]
- Rely on historical automotive pricing data. 
- DO NOT GUESS if the combination of features is impossible or you are completely uncertain.
- If the provided features are too sparse to confidently estimate a precise MSRP, you MUST output null.

[Vehicle Data]
* ModelYear: {ModelYear}
* Make: {Make}
* Model: {Model}
* Series: {Series}
* Trim: {Trim}
* FuelTypePrimary: {FuelTypePrimary}
* DriveType: {DriveType}
* EngineCylinders: {EngineCylinders}
* DisplacementL: {DisplacementL}
* TransmissionStyle: {TransmissionStyle}
* Doors: {Doors}

[Output Format]
Return ONLY a valid Dictionary object with a single key "estimated_msrp" containing the integer value of the MSRP. 
Success Example: {{"estimated_msrp": 38500}}
Uncertain/No Data Example: {{"estimated_msrp": null}}
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

class MSRPEstimate(BaseModel):
    estimated_msrp: int | None = None


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MSRP_MODEL = "gpt-4o-mini"


def estimate_msrp_from_features(row) -> int | None:
    """Return MSRP in USD only; None if uncertain or parsing fails. `row`: Series or mapping with VIN API columns."""

    # Data Preprocessing (Handling Empty, None)
    def _s(val):
        return "" if pd.isna(val) else str(val).strip()

    user_content = msrp_prompt_template.format(
        ModelYear=_s(row.get("ModelYear")),
        Make=_s(row.get("Make")),
        Model=_s(row.get("Model")),
        Series=_s(row.get("Series")),
        Trim=_s(row.get("Trim")),
        FuelTypePrimary=_s(row.get("FuelTypePrimary")),
        DriveType=_s(row.get("DriveType")),
        EngineCylinders=_s(row.get("EngineCylinders")),
        DisplacementL=_s(row.get("DisplacementL")),
        TransmissionStyle=_s(row.get("TransmissionStyle")),
        Doors=_s(row.get("Doors")),
    )
    completion = client.chat.completions.parse(
        model=MSRP_MODEL,
        messages=[{"role": "user", "content": user_content}],
        response_format=MSRPEstimate,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        return None
    return parsed.estimated_msrp

# Read VIN api Result file
vin_api_result = pd.read_csv("./Data/vin_api_result.csv")

# Create MRSP Feature using LLM
estimate_msrp = []
for _, row in vin_api_result.iterrows():
    # Check model Missing and IsFraud == True
    model_missing = pd.isna(row["Model"]) or str(row["Model"]).strip() == ""
    if model_missing or row["IsFraud"]:
        estimate_msrp.append(None)
    else:
        estimate_msrp.append(estimate_msrp_from_features(row))

vin_api_result["estimate_msrp"] = estimate_msrp
vin_api_result.to_csv("./Data/vin_api_result.csv", index=False, encoding="utf-8-sig")

# Merge estimate_msrp from vin_api_result into original_result
vin_api_result = pd.read_csv("./Data/vin_api_result.csv")
original_result = pd.read_csv("./Data/vehicles_10000.csv")

# Same row order as fetch_to_csv (matching row count and VIN order) — 1:1 assignment by row index
original_result["estimate_msrp"] = vin_api_result["estimate_msrp"].values

estimate_msrp_na_count = original_result["estimate_msrp"].isna().sum()
print(f"Number of missing estimate_msrp values: {estimate_msrp_na_count}")

original_result.head()

# Remove samples with missing values in the estimate_msrp column
original_result_noNa = original_result.dropna(subset=["estimate_msrp"])

# Remove rows where used price is higher than estimated MSRP
price_above_msrp_count = (original_result_noNa["price"] > original_result_noNa["estimate_msrp"]).sum()
print(f"price > estimate_msrp removed: {price_above_msrp_count}")
original_result_noNa = original_result_noNa[original_result_noNa["price"] <= original_result_noNa["estimate_msrp"]]

# Check final sample count
sample_count = original_result_noNa.shape[0]
print(f"Number of Data, original_result_noNa : {sample_count}")

# Save to vehicles_msrp.csv
original_result_noNa.to_csv("./Data/vehicles_msrp.csv", index=False, encoding="utf-8-sig")
