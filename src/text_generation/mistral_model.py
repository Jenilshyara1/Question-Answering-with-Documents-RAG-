from langchain_community.llms import LlamaCpp

class Chatbot:
    def __init__(self) -> None:
        self.template = """<s>[INST] You are an assistant for question-answering tasks.
        If you don't know the answer, just say that you don't know and return answer only nothing more.
        Use the bellow pieces of context to answer the question, if answer is not in context give appropriate answer by your own.
        [context] : {context}
        answer the question: {question}
        Answer: [/INST]"""
        self.llm = LlamaCpp(
            model_path=r"models/mistral-7b-instruct-v0.2.Q2_K.gguf",
            temperature=0.5,
            max_tokens=512,
            top_p=0.5,
            verbose=True,  # Verbose is required to pass to the callback manager
            n_ctx=512,
            n_threads=6,
            n_gpu_layers=33,
        )
        # self.chain = LLMChain(llm=self.llm, prompt=PromptTemplate(template=self.template, input_variables=["documents"]))

    def generate_response(self, prompt, context):
        # assistant_response = self.chain.run({"context": context, "question": prompt})
        assistant_response = self.llm(
            self.template.format(context=context, question=prompt)
        )
        return assistant_response


if __name__ == "__main__":
    chat = Chatbot()
    res = chat.generate_response(
        prompt="what is cheese making?",
        context="Cheesemaking is the process of turning milk into a semisolid mass. This is done by using a coagulating agent, such as rennet, acid, heat, or a combination of these",
    )
    print(res)