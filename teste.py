import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ==========================================
# CONFIG
# ==========================================
load_dotenv()

st.set_page_config(
    page_title="Plataforma de Entrevista",
    page_icon="🤖",
    layout="centered"
)

# ==========================================
# LLM
# ==========================================
@st.cache_resource
def init_llm_and_memory():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=500
    )

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=1000,
        return_messages=True
    )

    return llm, memory


llm, memory = init_llm_and_memory()

# ==========================================
# VALIDAÇÃO DE CARGO
# ==========================================
def is_cargo_valido(cargo: str):
    cargo = cargo.strip()

    if len(cargo) < 3:
        return False

    if not any(c.isalpha() for c in cargo):
        return False

    palavras = cargo.split()
    if len(palavras) == 1 and len(palavras[0]) < 4:
        return False

    return True

# ==========================================
# CLASSIFICAÇÃO DE VAGA (MELHORADA)
# ==========================================
def classificar_vaga(cargo: str):
    cargo = cargo.lower()

    tech = ["dev", "developer", "software", "backend", "frontend", "dados", "ti", "programador", "engenheiro"]
    estagio = ["estágio", "estagio", "trainee"]
    corporativo = ["marketing", "vendas", "financeiro", "rh", "administrativo", "analista"]
    operacional = ["garçom", "cozinheiro", "atendente", "motorista", "caixa"]

    if any(p in cargo for p in tech):
        return "tecnologia"
    elif any(p in cargo for p in estagio):
        return "estagio"
    elif any(p in cargo for p in corporativo):
        return "corporativo"
    elif any(p in cargo for p in operacional):
        return "operacional"
    else:
        return "geral"

# ==========================================
# PROMPTS (FORTE)
# ==========================================
def get_interview_prompt(cargo):
    categoria = classificar_vaga(cargo)

    system_template = f"""
    Você é um entrevistador especialista conduzindo uma entrevista profissional.

    Vaga: {cargo}
    Categoria: {categoria}

    ⚠️ REGRA PRINCIPAL:
    Você DEVE adaptar COMPLETAMENTE as perguntas para essa categoria.

    INSTRUÇÕES OBRIGATÓRIAS:

    Se categoria = tecnologia:
    - Faça perguntas técnicas (código, lógica, API, banco de dados)

    Se categoria = estagio:
    - Faça perguntas simples, comportamentais e de aprendizado

    Se categoria = corporativo:
    - Perguntas sobre comunicação, negócios e experiência

    Se categoria = operacional:
    - Perguntas práticas do dia a dia

    Se categoria = geral:
    - Perguntas genéricas de entrevista

    REGRAS:
    - Faça apenas 1 pergunta por vez
    - Sempre avalie a resposta antes da próxima pergunta
    - NÃO faça perguntas genéricas se houver categoria definida
    - Seja coerente com a vaga

    Comece com uma saudação + primeira pergunta.
    """

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])


def get_feedback_prompt():
    system_template = """
    Você é um Mentor de Carreira.

    IMPORTANTE:
    - Se o candidato respondeu menos de 2 perguntas → "Candidato inválido" e nota 0
    - Caso contrário → avaliação normal
    - Se as respostas forem totalmente fora de contexto, não pense duas vezes para zerar a nota do candidato

    Gere:
    - Pontos Fortes
    - Pontos a Melhorar
    - Nota (0 a 10)
    - Recomendações

    Use markdown.
    """

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

# ==========================================
# UI
# ==========================================
st.title("🤖 Plataforma de Entrevista")

if "fase" not in st.session_state:
    st.session_state.fase = "setup"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation" not in st.session_state:
    st.session_state.conversation = None

if "respostas_count" not in st.session_state:
    st.session_state.respostas_count = 0

if "encerrar_entrevista" not in st.session_state:
    st.session_state.encerrar_entrevista = False

# ==========================================
# SETUP
# ==========================================
if st.session_state.fase == "setup":

    st.markdown("### Prepare-se")

    cargo = st.text_input("Qual vaga?")

    if st.button("Iniciar") and cargo:

        if not is_cargo_valido(cargo):

            st.session_state.messages = [{
                "role": "assistant",
                "content": "❌ Candidato inválido.\n\n📊 Gerando feedback..."
            }]

            st.session_state.fase = "feedback"
            st.session_state.respostas_count = 0
            st.rerun()

        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=get_interview_prompt(cargo),
            verbose=False
        )

        st.session_state.fase = "entrevista"
        st.session_state.messages = [
            {"role": "system", "content": f"Vaga: {cargo}"}
        ]
        st.session_state.respostas_count = 0
        st.session_state.encerrar_entrevista = False

        st.rerun()

# ==========================================
# ENTREVISTA
# ==========================================
elif st.session_state.fase == "entrevista":

    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Primeira pergunta
    if len(st.session_state.messages) == 1:
        with st.chat_message("assistant"):
            resposta = st.session_state.conversation.predict(input="Iniciar")
            st.markdown(resposta)

            st.session_state.messages.append({
                "role": "assistant",
                "content": resposta
            })

    # Input
    if st.session_state.respostas_count < 4:
        user_input = st.chat_input("Digite sua resposta")

        if user_input:

            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })

            st.session_state.respostas_count += 1

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):

                if st.session_state.respostas_count >= 4:

                    resposta_final = """
✅ Entrevista Encerrada.

📊 Gerando Feedback...
                    """

                    st.markdown(resposta_final)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": resposta_final
                    })

                    st.session_state.encerrar_entrevista = True

                else:
                    resposta = st.session_state.conversation.predict(input=user_input)
                    st.markdown(resposta)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": resposta
                    })

    if st.session_state.encerrar_entrevista:
        st.session_state.encerrar_entrevista = False
        st.session_state.fase = "feedback"
        st.rerun()

# ==========================================
# FEEDBACK
# ==========================================
elif st.session_state.fase == "feedback":

    st.markdown("### 📊 Feedback")

    respostas = st.session_state.respostas_count

    if respostas < 2:
        st.markdown("""
### ❌ Candidato inválido

Nota: **0**

O candidato não respondeu perguntas suficientes para avaliação.
""")
    else:
        with st.spinner("Analisando sua entrevista..."):
            feedback_chain = ConversationChain(
                llm=llm,
                memory=memory,
                prompt=get_feedback_prompt()
            )

            relatorio = feedback_chain.predict(
                input=f"O candidato respondeu {respostas} perguntas. Gere o feedback."
            )

        st.markdown(relatorio)

    if st.button("Reiniciar"):
        st.session_state.clear()
        st.rerun()