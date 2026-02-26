from dataclasses import dataclass

INVALID_INPUT_RESPONSE = """The provided phone number contains non-numeric characters. 
Inform the user that they must provide a phone number using digits only to add WeChat.
Example:
    您的微信号手机号么？ 我这边只能加手机号呢。 
"""

INCOMPLETE_PHONE_NUM_RESPONSE = """The customer has not finished stating their phone number. 
Prompt them briefly to finish providing their phone number.**You should continue listen until full number.**
Example:
嗯嗯 # prompt the user to continue providing their phone number.
嗯，您继续 # prompt the user to continue providing their phone number.
"""

COMPLETE_PHONE_NUM_RESPONSE = """The user provided an 11-digit phone number: {phone_num}. 
Confirm with the user whether this number is correct. **Confirm the final number with user.**
Example:
    嗯，确认一下是 {phone_num}对么？
"""

LONG_NUMBER_RESPONSE = """The provided Phone number is longer than 11 digits.
Confirm with the user to provide a valid 11-digit phone number.
Example:
    {phone_num} 这个是手机号么？ 好像多了几位呢，您再确认一下哈？

"""

TRANSFER_TO_HUMAN_RESPONSE = """The system detected multiple invalid phone number inputs. 
Inform the user that adding WeChat is currently not possible and a human agent will follow up later.
Example:
    抱歉，我这边现在加不上您微信呢，稍后我们专员会联系您，添加个联系方式哈。
"""

TOOL_PROMPT = """Check the integrity of the provided wechat account(can be a phone number).
DO CHECK INTEGRITY IF THE CUSTOMER PROVIDES a wechat account (EACH TIME THE CUSTOMER TELLS A PARTIAL wechat account name, CALL THIS TOOL TO CHECK IT).        
Args:
    account_name (str): The wechat account name provided by the user. It should be a string of characters. (Full account name Collected so far, all available digits)
    Can also be partial input. The string should not include punctuations and spaces. 
    Remove spaces, hyphens, parentheses, commas, etc.
    Textual numbers should be converted to digits (the customer is intended to provide a number).
    Example: 1 for one, 2 for two, 3 for three...
    Example: 1 for "一", 2 for "二", 3 for "三"...
"""

@dataclass
class PhoneNumIntegrityChecker:

    call_count:int = 0
    max_call_count:int = 5
    current_phone_num_part = ""
    
    def preprocess(self, phone_num:str) -> str:
        """
        Preprocess the phone number.
        
        Args:
            phone_num (str): The phone number provided by the user.
        
        Returns:
            str: A string containing cleaned phone_number(space removed)
        """
        digits_map = {
            '零': '0',
            '一': '1',
            '二': '2',
            '三': '3',
            '四': '4',
            '五': '5',
            '六': '6',
            '七': '7',
            '八': '8',
            '九': '9',
            '幺': '1',
        }
        phone_num = str(phone_num).strip()
        for char, digit in digits_map.items():
            phone_num = phone_num.replace(char, digit)
        #  replace all spaces and punctuations
        phone_num = phone_num.replace(" ", "")
        return phone_num
    

    def check(
            self, 
            phone_num:str
        ) -> str:
        phone_num = self.preprocess(phone_num)
        is_digit = phone_num.isdigit()
        length = len(phone_num)
        
        self.current_phone_num_part = phone_num
        self.call_count += 1
        
        if self.call_count > self.max_call_count:
            return TRANSFER_TO_HUMAN_RESPONSE
        
        if not is_digit:
            return INVALID_INPUT_RESPONSE
        
        if length < 11:
            return INCOMPLETE_PHONE_NUM_RESPONSE
        
        if length == 11:
            return COMPLETE_PHONE_NUM_RESPONSE.format(phone_num=phone_num)
        
        return LONG_NUMBER_RESPONSE.format(phone_num=phone_num)
    
   
def create_phone_num_check_tool(
    max_call_count:int = 5, 
    tool_wrapper:callable = None
):
	"""
	Create a tool for each call.
	this tool keep track of the call count.
	Each tool instance can be used for one call session.
	"""
 
	checker = PhoneNumIntegrityChecker(max_call_count=max_call_count)

	def check_wechat_account_validity(
			account_name:str
		):
		return checker.check(account_name)

	check_wechat_account_validity.__doc__ = TOOL_PROMPT

	if tool_wrapper:
		return tool_wrapper(check_wechat_account_validity)

	return check_wechat_account_validity
