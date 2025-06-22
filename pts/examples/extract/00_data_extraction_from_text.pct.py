# %%
from magpy import *
import os
from datetime import datetime

# %% [markdown]
# Load API keys from the local `.env` file

# %%
from dotenv import load_dotenv
load_dotenv('.env', override=True)

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
# Simple example of extracting data from unstructured text. You specify a `schema` that contains the names of each field you want to extract and its type.

# %%
unstructured_text = """
As part of our ongoing review of political contributions, we have identified the following key donations:
On or around January 10, 2024, John Doe contributed $5,000 to the re-election campaign of Senator Smith.
This donation has been classified as an individual contribution.
Subsequently, on February 15, 2024, Jane Roe provided a donation of $10,000 to Governor Clark's campaign.
It is important to note that this contribution was made through a corporate entity.
Additionally, on March 5, 2024, Acme Corporation made a significant contribution in the amount of $50,000
to the Political Action Committee (PAC) associated with Mayor Johnson.
"""

schema = {
    "Donor Name": str,
    "Donation Amount": int,
    "Date": datetime,
    "Campaign Recipient": str,
    "Donation Type": str
}

extract_structured(
    text=unstructured_text,
    schema=schema,
)

# %% [markdown]
# Since we've configured *magpy* to cache all LLM calls, running `extract_structured` on the same text again with the same schema will simply retrieve the same result as before instantly. This is both to ensure replicability, but also to save costs.

# %%
extract_structured(
    text=unstructured_text,
    schema=schema,
)

# %% [markdown]
# You can use `magpy.extract.Field` to specify additional descriptions of the fields.

# %%
schema = {
    "Donor Name": Field(str, desc="The name of the donor"),
    "Donation Amount": Field(int, desc="The amount of the donation"),
    "Date": Field(datetime, desc="The date of the donation"),
    "Campaign Recipient": Field(str, desc="The recipient of the donation"),
    "Donation Type": Field(str, desc="The type of the donation")
}

extract_structured(
    text=unstructured_text,
    schema=schema,
)

# %% [markdown]
# You can specify that a field is optional, if it's not entirely certain that the unstructured data contains the relevant information.

# %%
schema = {
    "Donor Name": Field(str, desc="The name of the donor"),
    "Donor Address": Field(str, desc="The address of the donor", optional=True),
    "Donation Amount": Field(int, desc="The amount of the donation"),
    "Date": Field(datetime, desc="The date of the donation"),
    "Campaign Recipient": Field(str, desc="The recipient of the donation"),
    "Donation Type": Field(str, desc="The type of the donation"),
}

extract_structured(
    text=unstructured_text,
    schema=schema,
)

# %% [markdown]
# You can nest the schema for more flexible JSON-like data.

# %%
schema = {
    "Donor": Field({
        "Name": Field(str, desc="The name of the donor"),
        "Type": Field(str, desc="The type of the donor (e.g. company, individual, PAC, etc.)")
    }, desc="The donor of the donation"),
    "Recipient": Field({
        "Name": Field(str, desc="The name of the recipient"),
        "Type": Field(str, desc="The type of the recipient (e.g. company, individual, PAC, etc.)")
    }, desc="The recipient of the donation"),
    "Donation Amount": Field(int, desc="The amount of the donation"),
    "Date": Field(datetime, desc="The date of the donation"),
    "Donation Type": Field(str, desc="The type of the donation")
}

extract_structured(
    text=unstructured_text,
    schema=schema,
)
# Returns a dictionary of the structured data

# %% [markdown]
# You can extract date from multiple texts using the `extract_structured(texts=...)` argument.

# %%
unstructured_texts = [
    """
    As part of our ongoing review of political contributions, we have identified the following key donations:
    On or around January 10, 2024, John Doe contributed $5,000 to the re-election campaign of Senator Smith.
    This donation has been classified as an individual contribution.
    Subsequently, on February 15, 2024, Jane Roe provided a donation of $10,000 to Governor Clark's campaign.
    It is important to note that this contribution was made through a corporate entity.
    Additionally, on March 5, 2024, Acme Corporation made a significant contribution in the amount of $50,000
    to the Political Action Committee (PAC) associated with Mayor Johnson.
    """,
    
    """
    The following donations were identified as part of our ongoing review:
    On or around January 10, 2024, John Doe contributed $5,000 to the re-election campaign of Senator Smith.
    This donation has been classified as an individual contribution.
    Subsequently, on February 15, 2024, Jane Roe provided a donation of $10,000 to Governor Clark's campaign.
    It is important to note that this contribution was made through a corporate entity.
    Additionally, on March 5, 2024, Acme Corporation made a significant contribution in the amount of $50,000
    to the Political Action Committee (PAC) associated with Mayor Johnson.
    """,
    
    """
    In our recent analysis of political donations, we have documented the following contributions:
    On March 20, 2024, Emily White donated $2,500 to the campaign of Congressman Lee.
    This donation is categorized as an individual contribution.
    Furthermore, on April 10, 2024, Global Enterprises contributed $20,000 to the Senate campaign of Candidate Brown.
    This contribution was made through a corporate entity.
    Lastly, on May 1, 2024, the Community Fund donated $15,000 to the Political Action Committee (PAC) supporting Councilwoman Green.
    """
]

extract_structured(
    texts=unstructured_texts,
    schema=schema,
)
