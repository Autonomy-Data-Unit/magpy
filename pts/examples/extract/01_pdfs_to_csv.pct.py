# %%
from magpy import *
import os
from datetime import datetime

# %% [markdown]
# Configure the LLM

# %%
set_magpy_config(
    api_key=os.getenv("OPENAI_API_KEY"), 
    model_name="gpt-4o",    
    temperature=0.1,         
    cache_path='.cache'
)

# %% [markdown]
# Here we will extract data from annual accounts of UK charities. We'll use the below schema:

# %%
schema = {
    "Charity Details": Field({
        "Name": Field(str, desc="The name of the charity", optional=True),
        "Charity Number": Field(str, desc="The registered charity number", optional=True),
        "Year End": Field(datetime, desc="The financial year end date", optional=True),
    }, desc="Basic charity information", optional=True),
    
    "Financial Summary": Field({
        "Net Income": Field(float, desc="Net income before other recognised gains and losses", optional=True),
        "Investment Gains/Losses": Field(float, desc="Net gains or losses on investments", optional=True),
        "Accumulated Funds": Field(float, desc="Total accumulated funds at year end", optional=True),
        "Previous Year Accumulated Funds": Field(float, desc="Total accumulated funds from previous year", optional=True),
    }, desc="Key financial metrics", optional=True),
    
    "Income Sources": Field({
        "Subscriptions": Field(float, desc="Income from subscriptions", optional=True),
        "Investment Income": Field(float, desc="Income from investments", optional=True),
        "Miscellaneous Income": Field(float, desc="Other income sources", optional=True),
    }, desc="Breakdown of income sources", optional=True),
    
    "Expenditure": Field({
        "Grants": Field(float, desc="Total grants made", optional=True),
        "Maintenance": Field(float, desc="Maintenance and repair costs", optional=True),
        "Other Costs": Field(float, desc="Other operational costs", optional=True),
    }, desc="Breakdown of expenditure", optional=True),
    
    "Investments": Field({
        "Investment Manager": Field(str, desc="Name of the investment manager", optional=True),
        "Investment Fund": Field(str, desc="Name of the investment fund", optional=True),
        "Annual Return": Field(float, desc="Annual return percentage", optional=True),
        "Total Return Since Inception": Field(float, desc="Total return since fund inception", optional=True),
    }, desc="Investment details", optional=True),
    
    "Fixed Assets": Field({
        "Total Value": Field(float, desc="Total value of tangible fixed assets", optional=True),
        "Previous Year Value": Field(float, desc="Previous year's value of tangible fixed assets", optional=True),
    }, desc="Fixed assets information", optional=True),
    
    "Grant Making": Field({
        "Total Grants": Field(float, desc="Total amount of grants made", optional=True),
        "Number of Grants": Field(int, desc="Number of grants made", optional=True),
        "Grant Policy": Field(str, desc="Brief description of grant making policy", optional=True),
    }, desc="Grant making information", optional=True),
    
    "Activities": Field({
        "Main Objectives": Field(str, desc="Main charitable objectives", optional=True),
        "Key Achievements": Field(str, desc="Key achievements during the year", optional=True),
        "Future Plans": Field(str, desc="Plans for future periods", optional=True),
    }, desc="Charity activities and objectives", optional=True)
}

# %% [markdown]
# You can extract data from a single PDF like so

# %%
extract_structured(
    path='charity_accounts/5211224 2023-12-31 ROYAL ENGINEERS HEADQUARTER MESS.pdf',
    schema=schema,
)

# %% [markdown]
# *magpy* recognises several common textual file formats: `.pdf`, `.txt`, `.md`. If you are trying to extract text from an unsupported file format, you have to first convert it to a text format and use the `extract_structured(texts=...)` argument.

# %% [markdown]
# You can use `extract_structured(paths=...)` to extract texts from a list of files:

# %%
from pathlib import Path
pdfs = list(Path("charity_accounts/").glob("*.pdf"))
extract_structured(paths=pdfs, schema=schema)
