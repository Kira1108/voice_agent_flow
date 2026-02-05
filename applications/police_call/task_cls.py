from pydantic import BaseModel, Field

class PoliceCallBasicInfo(BaseModel):
    """Information collected during a police call. Need to collect all the inforamation before createing this object."""
    case_location: str = Field(..., description='The location of the case')
    case_type: str = Field(..., description='The type of the case')
    description: str = Field(..., description='The description of the case')
    caller_name: str = Field(..., description='The name of the caller')
    
    
    def transfer(self) -> str:
        print("Transfer to safety_suggestion")
        return 'safety_suggestion'
    
    
class SafetySuggestionProvided(BaseModel):
    """Base on collected information, provide safety suggestion to the caller. After the caller responded to your suggestion, create this object."""
    suggestion_provided:bool = Field(..., description='Whether safety suggestion is provided to the caller')
    
    
    def transfer(self) -> str:
        print("All tasks completed.")
        return 'end'