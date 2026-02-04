from pydantic import BaseModel, Field

class PartySizeResult(BaseModel):
    
    size: int = Field(..., description='The size of the party for the reservation')
    
    def transfer(self) -> str:
        print("Transferring to time_collector")
        return 'time_collector'
    
class TimeResult(BaseModel):
    
    time: str  =  Field(..., description='The time for the reservation in YYYY-MM-DD HH format')
    
    def check_availability(self, time_str:str) -> bool:
        # Here you would implement the actual logic to check availability
        # For demonstration purposes, let's assume all times are available
        return True
    
    def transfer(self) -> str:
        
        if self.check_availability(self.time):
            print("Reservation time is available. All tasks completed.")
            return 'end'
        
        else:
            print("Reservation time is not available. Please choose a different time.")
            return 'time_collector'
        