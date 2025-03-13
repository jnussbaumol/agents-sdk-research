from dotenv import load_dotenv
from os import getenv
from openai import AsyncAzureOpenAI
from agents import (
    Agent,
    Runner,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)
import asyncio


def setup() -> Agent:
    load_dotenv()
    endpoint = getenv("AZURE_OPENAI_ENDPOINT")
    if endpoint is None:
        # my LSP throws a fit about a "str | None" union if i don't check this
        print("missing endpoint")
        exit(1)

    # this block is required to use the Azure OpenAI service
    azure_openai_client = AsyncAzureOpenAI(
        api_key=getenv("AZURE_OPENAI_API_KEY"),
        api_version=getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=endpoint,
        azure_deployment=getenv("AZURE_OPENAI_DEPLOYMENT"),
    )
    set_tracing_disabled(disabled=True)
    set_default_openai_client(azure_openai_client)
    set_default_openai_api("chat_completions")

    product_expert = Agent(
        name="Product Expert",
        handoff_description="Specialist agent with full knowledge of a product's business use cases",
        instructions="You answer questions about the business logic of a product and assess the feasability of new features.",
    )
    software_expert = Agent(
        name="Software Expert",
        handoff_description="Specialist agent with deep technical knowledge of a product's codebase",
        instructions="You explain technical details about a product and assess the feasability of new features.",
    )
    data_expert = Agent(
        name="Data Expert",
        handoff_description="Specialist agent with deep technical knowledge of a product's database",
        instructions="You explain technical details about the data and database for a product and assess the feasability of new features.",
    )
    # we don't need to send any other agents to main(), thanks to the handoff parameter.
    orchestrator = Agent(
        name="Orchestrator",
        instructions="You determine which agent to use based on the user's question",
        model="gpt-4o",
        handoffs=[product_expert, software_expert, data_expert],
    )
    return orchestrator


async def main(orchestrator: Agent) -> None:
    result = await Runner.run(
        orchestrator,
        "I'd like to add a button that, when pressed, displays all currently active users and their monthly averages for time and money spent on the app.",
    )
    print(result.final_output)


if __name__ == "__main__":
    orchestrator = setup()
    asyncio.run(main(orchestrator))
