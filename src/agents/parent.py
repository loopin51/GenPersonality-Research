from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from src.config import (
    PARENT_MODEL_NAME, 
    OPENROUTER_API_KEY, 
    LLM_BASE_URL, 
    SITE_URL, 
    SITE_NAME
)
import os

class ParentAgent:
    def __init__(self, persona_type: str):
        """
        persona_type: 'Warm' or 'Cold'
        """
        self.persona_type = persona_type
        self.llm = ChatOpenAI(
            model=PARENT_MODEL_NAME, 
            temperature=0.7, 
            api_key=OPENROUTER_API_KEY,
            base_url=LLM_BASE_URL,
            default_headers={
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_NAME,
            }
        )

    def respond(self, context: str, child_text: str) -> str:
        """
        Generates parent response.
        """
        # Load System Prompt
        from prompts.templates import SYSTEM_PROMPT_PARENT_WARM, SYSTEM_PROMPT_PARENT_COLD, PARENT_RESPOND_PROMPT
        
        if self.persona_type == "Warm":
            system_prompt_text = SYSTEM_PROMPT_PARENT_WARM
        elif self.persona_type == "Cold":
            system_prompt_text = SYSTEM_PROMPT_PARENT_COLD
        else:
            system_prompt_text = f"You are a {self.persona_type} parent. Act accordingly."

        # Load Chat Prompt
        human_template = PARENT_RESPOND_PROMPT.format(
            context=context,
            child_text=child_text
        )
        
        # We don't use ChatPromptTemplate with placeholders because it is already formatted.
        # But we need to pass Messages to invoke.
        from langchain.schema import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_prompt_text),
            HumanMessage(content=human_template)
        ]
        
        # Define JSON Schema
        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "parent_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "internal_thought": {
                            "type": "string",
                            "description": "Analysis of child's behavior based on persona"
                        },
                        "response_speech": {
                            "type": "string",
                            "description": "Actual speech to the child in Korean"
                        }
                    },
                    "required": ["internal_thought", "response_speech"],
                    "additionalProperties": False
                }
            }
        }

        try:
            # Bind max_tokens to prevent infinite loops (costly!)
            structured_llm = self.llm.bind(response_format=json_schema, max_tokens=4096)
            response = structured_llm.invoke(messages)
            import json
            cleaned_response = response.content.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned_response)
            speech = data.get("response_speech", response.content)
        except Exception as e:
            print(f"Parent Generation Error: {e}")
            print(f"Raw Response causing error: {response.content}")
            speech = "(부모가 침묵합니다.)" # Safe fallback

        return speech
