import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ["OPENAI_API_KEY"] = "sk-test"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["PYTHON_DOTENV_DISABLED"] = "true"

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from gtm_agent import gtm_agent as agent_module
from run import EXAMPLES


class ScriptedModel(BaseChatModel):
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "scripted-test-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        request = next(message.content for message in messages if message.type == "human")
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        last_tool = tool_messages[-1].name if tool_messages else None

        if last_tool is None:
            response = self._tool_call("get_current_rep", {})
        elif "email" in request.lower():
            response = self._email_response(last_tool, tool_messages)
        else:
            response = self._score_response(last_tool, tool_messages)

        return ChatResult(generations=[ChatGeneration(message=response)])

    def _email_response(self, last_tool, tool_messages):
        if last_tool == "get_current_rep":
            return self._tool_call("get_prospect", {"prospect_id": "LEAD-12853"})
        if last_tool == "get_prospect":
            prospect = self._result(tool_messages, "get_prospect")["prospect"]
            return self._tool_call("send_prospect_email", {
                "prospect": prospect,
                "subject": "Scheduling your demo",
                "body": "Let's schedule your demo for next week.",
            })
        return AIMessage(content="The email was sent.")

    def _score_response(self, last_tool, tool_messages):
        if last_tool == "get_current_rep":
            return self._tool_call("update_prospect_info", {
                "prospect_id": "LEAD-71001",
                "technology": "Kafka",
            })
        if last_tool == "update_prospect_info":
            return self._tool_call("lookup_offering", {"offering_id": "OFFER-10005"})
        if last_tool == "lookup_offering":
            return self._tool_call("build_prospect_profile", {"prospect_id": "LEAD-71001"})
        if last_tool == "build_prospect_profile":
            return self._tool_call("score_prospect", {
                "prospect_profile": self._result(tool_messages, "build_prospect_profile")["prospect_profile"],
                "offering": self._result(tool_messages, "lookup_offering")["offering"],
            })
        return AIMessage(content="The prospect was updated and scored.")

    @staticmethod
    def _result(tool_messages, name):
        content = next(message.content for message in reversed(tool_messages) if message.name == name)
        return json.loads(content)

    @staticmethod
    def _tool_call(name, args):
        return AIMessage(content="", tool_calls=[{
            "name": name,
            "args": args,
            "id": f"call-{name}",
            "type": "tool_call",
        }])


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.model = ScriptedModel()
        self.agent = create_agent(
            model=self.model,
            tools=[
                agent_module.lookup_offering,
                agent_module.build_prospect_profile,
                agent_module.get_prospect,
                agent_module.send_prospect_email,
                agent_module.score_prospect,
                agent_module.update_prospect_info,
                agent_module.get_current_rep,
            ],
            system_prompt=agent_module.SYSTEM_PROMPT,
        )
        self.original_agent = agent_module.gtm_agent
        agent_module.gtm_agent = self.agent

    def tearDown(self):
        agent_module.gtm_agent = self.original_agent

    def test_run_example_sends_email(self):
        user_id, request = EXAMPLES[0]
        result = agent_module.run_agent(request, user_id=user_id)

        self.assertIn("sent", result["reply"])
        tool_messages = [message for message in self.model.calls[-1] if isinstance(message, ToolMessage)]
        self.assertEqual(
            [message.name for message in tool_messages],
            ["get_current_rep", "get_prospect", "send_prospect_email"],
        )
        self.assertEqual(json.loads(tool_messages[-1].content)["status"], "sent")
        self.assertEqual(self.model.calls[0][0].content, agent_module.SYSTEM_PROMPT)

    def test_run_example_updates_and_scores_prospect(self):
        user_id, request = EXAMPLES[3]
        score = SimpleNamespace(model_dump=lambda: {"score": 82, "justification": "Good fit."})
        with patch.object(agent_module, "_scoring_llm", SimpleNamespace(invoke=lambda messages: score)):
            result = agent_module.run_agent(request, user_id=user_id)

        self.assertIn("updated and scored", result["reply"])
        tool_messages = [message for message in self.model.calls[-1] if isinstance(message, ToolMessage)]
        self.assertEqual(
            [message.name for message in tool_messages],
            [
                "get_current_rep",
                "update_prospect_info",
                "lookup_offering",
                "build_prospect_profile",
                "score_prospect",
            ],
        )
        self.assertEqual(json.loads(tool_messages[-1].content)["score"], 82)
        self.assertEqual(self.model.calls[0][0].content, agent_module.SYSTEM_PROMPT)

    def test_agent_exposes_only_gtm_tools(self):
        expected_tools = {
            "lookup_offering",
            "build_prospect_profile",
            "get_prospect",
            "send_prospect_email",
            "score_prospect",
            "update_prospect_info",
            "get_current_rep",
        }

        self.assertEqual(set(self.original_agent.nodes["tools"].bound.tools_by_name), expected_tools)

    def test_first_model_call_uses_only_application_prompt(self):
        agent_module.run_agent(EXAMPLES[0][1], user_id=EXAMPLES[0][0])
        system_message = self.model.calls[0][0]

        self.assertEqual(system_message.content, agent_module.SYSTEM_PROMPT)
        for harness_term in ("todo", "filesystem", "subagent", "task tool"):
            self.assertNotIn(harness_term, system_message.content.lower())


if __name__ == "__main__":
    unittest.main()
