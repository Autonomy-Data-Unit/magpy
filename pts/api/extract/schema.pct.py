# %% [markdown]
# # schema

# %%
#|default_exp extract.schema

# %%
#|hide
import nblite; from nbdev.showdoc import show_doc; nblite.nbl_export()

# %%
#|export
from typing import Union, Dict, Any, Type, get_args, get_origin, List, Optional
from datetime import datetime
import pydantic
import json
from pydantic import BaseModel, Field as PydanticField, create_model


# %%
#|export
class Field:
    """Schema field definition with type enforcement, descriptions, and optional fields."""
    
    def __init__(
        self, 
        field_type: Any,
        desc: Optional[str] = None, 
        optional: bool = False
    ):
        self.field_type = field_type
        self.desc = desc
        self.optional = optional


# %%
#|exporti
def _convert_field_to_type(field: Field) -> tuple:
    """
    Converts a custom Field into a (type_annotation, FieldInfo) tuple for Pydantic
    """
    desc = field.desc or ""
    annotation = _parse_type(field.field_type)
    if field.optional:
        annotation = Optional[annotation]
    return (annotation, PydanticField(... if not field.optional else None, description=desc))

def _parse_type(field_type: Union[Type, Dict, list, tuple]) -> Any:
    """
    Recursively resolve field_type to proper typing annotations for Pydantic
    """
    if isinstance(field_type, dict):
        # Nested object -> create a dynamic Pydantic model
        return _create_dynamic_model(field_type)
    
    origin = get_origin(field_type)
    if origin is list or origin is List:
        (item_type,) = get_args(field_type)
        return List[_parse_type(item_type)]
    
    return field_type  # e.g., str, int, datetime

def _create_dynamic_model(schema_dict: Dict[str, Field], model_name: str = "ResponseModel") -> Type[BaseModel]:
    """
    Create a Pydantic model from a dictionary of Field instances
    """
    fields = {}
    for field_name, field_def in schema_dict.items():
        if not isinstance(field_def, Field): # if the field is not a Field, then we presume it is a type
            field_def = Field(field_def)
        fields[field_name] = _convert_field_to_type(field_def)

    model = create_model(model_name, **fields)
    
    # Monkey-patch __config__ to disable additional properties
    class Config:
        extra = "forbid"
    model.__config__ = Config
    return model


# %%
#|exporti
from openai.types.chat.completion_create_params import ResponseFormat as OpenAIResponseFormatParam

def _pydantic_to_response_format_param(
    model: Type[pydantic.BaseModel] | Type[Any]
) -> OpenAIResponseFormatParam:
    """
    Convert a Pydantic model to OpenAI's response_format parameter.
    
    This function returns the complete response_format parameter structure
    that can be passed directly to the OpenAI API.
    
    Args:
        model: A Pydantic model class (BaseModel or dataclass-like)
        
    Returns:
        OpenAIResponseFormatParam: The complete response_format parameter
        
    Raises:
        TypeError: If the model type is not supported
    """
    from openai.lib._parsing import type_to_response_format_param
    from openai._types import NOT_GIVEN
    
    response_format = type_to_response_format_param(model)
    
    if response_format is NOT_GIVEN:
        raise TypeError(f"Could not convert model {model} to response format")
    
    return response_format


# %%
#|hide
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

dynamic_model = _create_dynamic_model(schema)
openai_response_format = _pydantic_to_response_format_param(dynamic_model)

import adulib.llm
from adulib.caching import set_default_cache_path

prompt = """
As part of our ongoing review of political contributions, we have identified the following key donations:
On or around January 10, 2024, John Doe contributed $5,000 to the re-election campaign of Senator Smith.
This donation has been classified as an individual contribution.
Subsequently, on February 15, 2024, Jane Roe provided a donation of $10,000 to Governor Clark's campaign.
It is important to note that this contribution was made through a corporate entity.
Additionally, on March 5, 2024, Acme Corporation made a significant contribution in the amount of $50,000
to the Political Action Committee (PAC) associated with Mayor Johnson.
"""

set_default_cache_path(".cache")
res = adulib.llm.single(
    prompt=prompt,
    response_format=openai_response_format,
    model="gpt-4o-mini"
)

dynamic_model.model_validate(json.loads(res))
