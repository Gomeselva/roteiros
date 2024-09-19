from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from crewai_tools import SerperDevTool, DallETool
from langchain_openai import ChatOpenAI
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

gpt4o = ChatOpenAI(model_name="gpt-4o-mini")

search_tool = SerperDevTool()
dalle_tool = DallETool()

# 1. Agente Pesquisador do tema

pesquisador = Agent(
    role = "Pesquisador de Tema",
    goal = "Pesquisar informações detalhadas sobre o {tema} na internat.",
    backstory=(
        "Você é um especialista, com habilidades aguçadas para "
        "encontrar informações valiosas e detalhadas na internet."
    ),
    verbose=True,
    memory=True,
    tools=[search_tool],
    llm=gpt4o,
)

# 2. Agente Escritor de Títulos
escritor_titulos = Agent(
    role = "Escritor de Títulos de Vídeos",
    goal = "Criar Títulos atraentes e otimizados para vídeos sobre {tema}",
    verbose=True,
    memory=True,
    backstory=(
        "Vocé tem uma habilidade especial para criar títulos que capturam a "
        "essência de um vídeo e atraem a atençāo do público."
),
    llm=gpt4o
)

# 3. Agente Escritor de Roteiros
escritor_roteiro = Agent(
    role = "Escritor de Roteiro",
    goal = "Escrever um roteiro detalhado e envolvente para um vídeo sobre {tema}",
    verbose=True,
    memory=True,
    backstory=
        f"""Você é um contador de histórias talentoso, capaz de transformar 
        informaçōes em narrativas criativas para vídeos."""
    ,
    llm=gpt4o
)

# 4. Agente Especialista em SEO para YouTube
especialisat_seo = Agent(
    role = "Especialista em SEO para YouTube",
    goal = "Otimizar o roteiro e o título para que o vídeo tenha alta performance no YouTube",
    verbose=True,
    memory=True,
    backstory=(
        "Você é um expert em SEO, com profundo entendimento das melhores "
        "práticas para otimizar conteúdo para o YouTube."
    ),
    llm=gpt4o
)

# 5. Agente Criador de Prompts para DALL-E
criador_prompt_dalle = Agent(
    role = "Criador de Prompts para DALL-E",
    goal = "Escrever um prompt para gerar uma imagem usando DALLE-E com base no tema do vídeo {tema}",
    verbose=True,
    memory=True,
    backstory=(
        "Você é especialista em criar descrições detalhadas e imaginativas que "
        "permitem ao DALL-E gerar imagens impressionantes com base em textos."
    ),
    llm=gpt4o
)

# 6. Agente Gerador de Imagens com DALL-E
gerador_imagens = Agent(
    role = "Gerador de Imagens com DALL-E",
    goal = "Gerar uma imagem usando DALL-E como o prompt fornecido pelo criador de Prompts para DALL-E sobre {tema}",
    verbose=True,
    memory=True,
    backstory=(
        "Você é um mestre em gerar imagens incríveis com o DALL-E, "
        "capaz de transformar descrições em obras de arte visuais."
    ),
    tools=[dalle_tool],
    llm=gpt4o
)


# 7. Agente Revisor de Conteúdo 
revisor = Agent(
    role = "Revisor de Conteúdo",
    goal = "Revisar todo conteúdo produzido, incluir links das imagens geradas e entrgar a versão final do vídeo",
    verbose=True,
    memory=True,
    backstory=(
        "Você é um perfeccionista, com olhos afiados para detalhes e uma "
        "habilidade especial para garantir a qualidade do conteúdo."
    ),
    llm=gpt4o
)

# 1. Tarefa Pesquisa
tarefe_pesquisa = Task(
    description=(
        "Pesquisar informações detalhadas e relevantes sobre o tema: {tema}. "
        "Concentre-se em aspectos únicos e dados importantes que podem enriquecer o vídeo, "
        "Todo o texto deve estar em Português Brasil."
    ),
    agent=pesquisador,
    output_file="pesquisa.md",
    expected_output="Informações detalhadas e relevantes sobre o {tema}."
    )
# 2. Tarefa de Esrita de Títulos
tarefa_titulos = Task(
    description=(
        "Criar títulos atraentes e otimizados para vídeos sobre o tema: {tema}. "    
        "Certifique-se de que o título seja cativante e esteja otimizado para SEO. "
        "Todo o texto deve estar em Português Brasil."
    ),
    expected_output="Um título otimizado para vídeo sobre {tema}.",
    agent=escritor_titulos,
)

# 3. Tarefa de Escrita de Roteiro
tarefe_roteiro = Task(
    description=(
        "Escrever um roteiro detalhado e envolvente para um vídeo sobre o tema: {tema}. "
        "O roteiro deve ser envolvente e fornecer um fluxo lógico de informações. "
        "Se necessário, especifique imagens que podem enriquecer o conteúdo. "
        "Todo o texto deve estar em Português Brasil."
    ),
    expected_output="Um roteiro completo e bem estrtuturado para o vídeo sobre o {tema}.",
    agent=escritor_roteiro,
)

# 4. Tarefa de Otimização de SEO
tarefa_seo = Task(
    description=(
        "Otimizar o roteiro e o título para que o vídeo tenha alta performance no YouTube. "
        "Incorporar as melhores práticas de SEO para garantir uma boa classificação e visibilidade. "
        "Crie hashtags, palavras-chaves e tags de víeos para youtube. "
        "Todo o texto deve estar em Português Brasil."
    ),
    expected_output="Um roteiro e título otimizados para o vídeo sobre o {tema}, prontos para publicação.",
    agent=especialisat_seo,
)

# 5. Tarefa de Criação de Prompt para DALL-E
tarefa_criacao_prompt_dalle = Task(
    description=(
        "Criar um prompt etalhado para gerar uma imagem no DALL-E "
        "conforme descrito no roteiro. "
        "Todo o texto deve estar em Português Brasil."
    ),
    expected_output="Um prompt criativo e detalhado para gerar uma imagem com DALL-E sobre o {tema}.",
    agent=criador_prompt_dalle,
)

# 6. Tarefa de Geração de Imagem com DALL-E
tarefa_geracao_imagem = Task(
    description=(
        "Gerar uma imagem usando DALL-E com o prompt fornecido pelo Criador de Prompts para DALL-E. "
    ),
    expected_output="Uma imagem gerada pronta para uso.",
    agent=gerador_imagens,
)

#7. Tarefa de Revisão de Conteúdo
tarefa_revisao = Task(
    description=(
        "Revisar todo o conteúdo produzido (título, roteiro, e otimização de SEO). "
        "incluir os links das imagens geradas e preparar a versão final para entrega ao usuário."
        "Todo o texto deve estar em Português Brasil."
    ),
    expected_output="Conteúdo revisado, com links das imagens inclusos, pronto para entrega ao usuário.",
    agent=revisor,
    output_file="conteudo_final.md"
)

# Formando a crew
crew = Crew(
    agents=[pesquisador, escritor_titulos, escritor_roteiro, especialisat_seo, criador_prompt_dalle, gerador_imagens, revisor],
    tasks=[tarefe_pesquisa, tarefa_titulos, tarefe_roteiro, tarefa_seo, tarefa_criacao_prompt_dalle, tarefa_geracao_imagem, tarefa_revisao],
    process=Process.sequential
)

result = crew.kickoff(inputs={"tema":"Veículos Autônomos em Defesa e Segurança"})