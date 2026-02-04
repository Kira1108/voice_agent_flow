from pydantic import BaseModel, Field

class CustomerName(BaseModel):
    """Greet the customer with the customer's name in the greeting message. If not explicitly told that the name is wrong or wrong name dialed, assume the name is confirmed.
    Example: 
        Customer Service Representative: "喂，您好，请问是xxx吗?"
        Customer: "/呃/你说/是的/哪里/咋了/哦/是/嗯...."  -> create CustomerName(name="xxx") # all the ambiguous response should be treated as confirmation
        Customer: "/不是/打错了/我不是" -> do not create
    """
    name: str = Field(..., description = 'The name of the customer')
    
    
    def transfer(self) -> str:
        print("Transferring to FinancialSupportStatus")
        return 'financial_support_inquiry'
    
    
class FinancialSupportStatus(BaseModel):
    """Whether the customer need financial support. True is the customer do not refuse/reject financial support.
    As long the the customer asking product details or show insterest, create FinancialSupportInquiry(require_support=True).
    Explicit rejection or refusal should create FinancialSupportInquiry(require_support=False).
    Other case, you should continue to ask the customer for clarification.
    """
    require_support: bool = Field(..., description = 'Whether the customer need financial support')
    
    def transfer(self) -> str:
        print("Transferring to VehicleNotUnderRepayment")
        return 'vehicle_payment_status'
    
    
class VehicleNotUnderRepayment(BaseModel):
    """Whether the vehicle is not under repayment.
    If the vehicle is fully paid off, create VehicleNotUnderRepayment(is_not_under_repayment=True).
    If the vehicle is still under repayment, create VehicleNotUnderRepayment(is_not_under_repayment=False).
    """
    is_not_under_repayment: bool = Field(..., description = 'Whether the vehicle is not under repayment')
    
    
    def transfer(self) -> str:
        print("Transferring to VehicleLiscenceUnderControl")
        return 'vehicle_liscence_under_control'
    
class VehicleLiscenceUnderControl(BaseModel):
    """Whether the vehicle liscence(Chinese机动车行驶证， 绿本， 大本) is under the customer's control.
    If the vehicle liscense is under the customer's control, create VehicleLiscenceUnderControl(is_under_control=False).
    If the vehicle liscense is under the customer's spouse or family member's control, create VehicleLiscenceUnderControl(is_under_control=True).
    If the vehicle liscense ir under his/her company's control, create VehicleLiscenceUnderControl(is_under_control=False).
    """
    
    is_under_control: bool = Field(..., description = 'Whether the vehicle liscence is under the customer\'s control')
    
    def transfer(self) -> str:
        return "agree_wechat_add"


class AgreeToAddWechatAccount(BaseModel):
    """Whether the customer agree to add wechat account for further contact.
    If the customer agree to add wechat account, create AgreeToAddWechatAccount(agree=True).
    If the customer refuse to add wechat account, create AgreeToAddWechatAccount(agree=False).
    """
    agree: bool = Field(..., description = 'Whether the customer agree to add wechat account')
    
    
    def transfer(self) -> str:
        return "ask_wechat_id"
    
    
class WeChatId(BaseModel):
    """The WeChat ID provided by the customer for further contact.
    Example:
        Customer Service Representative: "请问您的微信号是多少？"
        Customer: "我的微信号是abc123" -> create WeChatId(wechat_id="abc123")
    """
    wechat_id: str = Field(..., description = 'The WeChat ID provided by the customer')
    
    
    def transfer(self) -> str:
        print("All tasks completed.")
        return 'end'